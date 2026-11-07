"""Matched identity-draft / real-draft LoRA experiment; portable historical training loop."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import pathlib,json,random,hashlib,time,fcntl,argparse
p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=42);p.add_argument("--run-name",default="input-grounded-repair-v1");p.add_argument('--base',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args()
R=pathlib.Path(__file__).resolve().parents[1];D=pathlib.Path(a.data);OUT=pathlib.Path(a.output);BASE=pathlib.Path(a.base)
datasets={arm:[json.loads(l) for l in (D/f'{arm}.jsonl').open()] for arm in ['identity','repair']}
assert [x['id'] for x in datasets['identity']]==[x['id'] for x in datasets['repair']]
assert [x['targets'] for x in datasets['identity']]==[x['targets'] for x in datasets['repair']]
rng=random.Random(a.seed);batches=[[rng.randrange(len(datasets['identity'])) for _ in range(16)] for step in range(64)]
plan={'seed':a.seed,'steps':64,'effective_batch':16,'micro_batch':4,'lr':5e-5,'lora_rank':16,'examples':len(datasets['identity']),'training_manifest':json.loads((D/'manifest.json').read_text()),'batch_plan_sha256':hashlib.sha256(json.dumps(batches).encode()).hexdigest(),'factor':'identity-gold draft vs real parent greedy draft only','validation_selection':'No hyperparameter selection; fixed64steps; evaluate dev then freeze both for blind transfer'}
if a.preflight:
 print(json.dumps(plan,indent=2));raise SystemExit(0)
if OUT.exists() and any(OUT.iterdir()):raise FileExistsError('Use a fresh output directory')
OUT.mkdir(parents=True)
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import get_peft_model,LoraConfig
(R/'.cache').mkdir(exist_ok=True)
(OUT/'preregistered-plan.json').write_text(json.dumps(plan,indent=2))
with (R/'.cache/gpu.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 tokenizer=AutoTokenizer.from_pretrained(BASE);tokenizer.pad_token=tokenizer.eos_token
 for arm in ['identity','repair']:
  rows=datasets[arm];data=[(tokenizer.encode(x['inputs'],add_special_tokens=False),tokenizer.encode(x['targets'],add_special_tokens=False)) for x in rows]
  folder=OUT/arm;folder.mkdir(exist_ok=True)
  if (folder/'COMPLETE').exists():continue
  torch.manual_seed(a.seed);model=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
  model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False;model.train()
  initial=hashlib.sha256(b''.join(p.detach().cpu().float().numpy().tobytes() for n,p in model.named_parameters() if 'lora_' in n)).hexdigest()
  optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-5);metrics=[];started=time.monotonic()
  for step,indices in enumerate(batches):
   optimizer.zero_grad();loss_sum=0.;tokens=0
   for off in range(0,16,4):
    ii=indices[off:off+4];length=max(sum(map(len,data[i])) for i in ii);ids=[];labels=[];masks=[];weights=[]
    for i in ii:
     prompt,target=data[i];pad=length-len(prompt)-len(target);ids.append(prompt+target+[tokenizer.pad_token_id]*pad);labels.append([-100]*len(prompt)+target+[-100]*pad);masks.append([1]*(len(prompt)+len(target))+[0]*pad)
     weights.append(1.);tokens+=len(target)
    ids=torch.tensor(ids,device='cuda');labels=torch.tensor(labels,device='cuda');mask=torch.tensor(masks,device='cuda')
    logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits[:,:-1].float();target=labels[:,1:];valid=target!=-100
    per=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),target.reshape(-1),reduction='none',ignore_index=-100).reshape(target.shape)
    per=(per*valid).sum(1)/valid.sum(1);loss=(per*torch.tensor(weights,device='cuda')).sum()/16;loss.backward();loss_sum+=float(loss.detach())
    del logits,per,loss
   grad=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.));optimizer.step()
   metric={'step':step+1,'loss':loss_sum,'grad_norm':grad,'target_tokens':tokens,'input_tokens':sum(len(data[i][0]) for i in indices)};metrics.append(metric)
   (folder/'metrics.json').write_text(json.dumps(metrics,indent=2));print(arm,json.dumps(metric),flush=True)
  model.save_pretrained(folder/'final-adapter');tokenizer.save_pretrained(folder/'final-adapter')
  (folder/'run.json').write_text(json.dumps({**plan,'arm':arm,'initial_lora_sha256':initial,'elapsed_seconds':time.monotonic()-started,'base_model':str(BASE),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'assistant_target_only':True},indent=2));(folder/'COMPLETE').touch()
  del model,optimizer;torch.cuda.empty_cache()
