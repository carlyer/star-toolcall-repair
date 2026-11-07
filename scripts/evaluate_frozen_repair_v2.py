"""Actual parent+repair cascade; labels are consulted only after selecting output."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0');os.environ.setdefault('VLLM_PLUGINS','')
import argparse,pathlib,json,sys,fcntl,time,collections,hashlib
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import importlib.util
_gate_spec=importlib.util.spec_from_file_location('frozen_edited_gate',R/'src/schema_gate.py')
_gate=importlib.util.module_from_spec(_gate_spec);_gate_spec.loader.exec_module(_gate)
parse_calls=_gate.parse_calls;grounded_refine_gate=_gate.structured_refine_gate
from input_grounded_prompt import make_prompt
from fixed_scoring import score,summary
from vllm import LLM,SamplingParams
from vllm.lora.request import LoRARequest

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--control',required=True);p.add_argument('--repair',required=True);p.add_argument('--parent-cache');p.add_argument('--self-cache');p.add_argument('--phase',choices=['dev','blind'],required=True);p.add_argument('--frozen-method');a=p.parse_args()
 if a.phase=='blind':
  protocol=json.loads(pathlib.Path(a.frozen_method).read_text()) if a.frozen_method else {}
  assert protocol.get('test_unlocked') is True,'Blind stays locked pending final method selection'
  assert hashlib.sha256(pathlib.Path(a.data).read_bytes()).hexdigest()==protocol['blind_data_sha256']
  assert hashlib.sha256((R/'src/schema_gate.py').read_bytes()).hexdigest()==protocol['gate_sha256']
 out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE').exists():raise RuntimeError('Do not overwrite completed cascade')
 tasks=[json.loads(l) for l in pathlib.Path(a.data).open()];base=pathlib.Path(a.base);started=time.monotonic()
 (R/'.cache').mkdir(exist_ok=True)
 with (R/'.cache/gpu.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  llm=LLM(model=str(base),dtype='bfloat16',tensor_parallel_size=1,gpu_memory_utilization=.65,max_model_len=6656,enforce_eager=True,enable_lora=True,max_lora_rank=16)
  params=SamplingParams(temperature=0,max_tokens=512)
  if a.parent_cache:
   cache={x['id']:x for x in map(json.loads,pathlib.Path(a.parent_cache).open())};parents=[cache[x['id']] for x in tasks]
   assert len({x['id'] for x in tasks})==len(tasks)
   for task,parent in zip(tasks,parents):assert parent['prompt_sha256']==hashlib.sha256(task['inputs'].encode()).hexdigest(),task['id']
  else:
   generated=llm.generate([x['inputs'] for x in tasks],params);parents=[score(x,y.outputs[0].text,y.outputs[0].finish_reason) for x,y in zip(tasks,generated)]
  (out/'parent.predictions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in parents))
  queries=[];indices=[]
  for i,(task,parent) in enumerate(zip(tasks,parents)):
   try:calls=parse_calls(parent['output'])
   except (ValueError,KeyError,TypeError):continue
   if calls:queries.append(make_prompt(task['inputs'],calls));indices.append(i)
  reports={'parent':summary(parents)}
  for arm,adapter,lora_id in [('parent-reask',None,0),('identity',a.control,1),('repair',a.repair,2)]:
   if arm=='parent-reask' and a.self_cache:
    source={x['id']:x for x in map(json.loads,pathlib.Path(a.self_cache).open())};records=[source[x['id']] for x in tasks];reports[arm]=summary(records)
    (out/f'{arm}.predictions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records));continue
   req=LoRARequest(arm,lora_id,adapter) if adapter else None;generated=llm.generate(queries,params,lora_request=req) if queries else []
   records=list(parents);audits=[];tokens=0
   for i,generation in zip(indices,generated):
    text='</think>\n'+generation.outputs[0].text;chosen,audit=grounded_refine_gate(tasks[i]['inputs'],parents[i]['output'],text)
    record=score(tasks[i],chosen,generation.outputs[0].finish_reason if audit['accepted'] else parents[i]['finish_reason']);record['gate']=audit;records[i]=record;tokens+=len(generation.outputs[0].token_ids)
    audits.append({'id':tasks[i]['id'],'gate':audit,'parent_output':parents[i]['output'],'refinement_output':text,'tokens':len(generation.outputs[0].token_ids)})
   reports[arm]={**summary(records),'extra_generation_calls':len(indices),'extra_generated_tokens':tokens,'gate_counts':dict(collections.Counter(x['gate']['reason'] for x in audits))}
   (out/f'{arm}.predictions.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records));(out/f'{arm}.gate.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in audits))
  (out/'summary.json').write_text(json.dumps({'phase':a.phase,'data':a.data,'control':a.control,'repair':a.repair,'parent_cache':a.parent_cache,'self_cache':a.self_cache,'elapsed_seconds':time.monotonic()-started,'metrics':reports,'labels_in_gate':False,'additional_token_cap_per_candidate':512,'parent_no_call_preserved':True},indent=2));(out/'COMPLETE').touch();print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
