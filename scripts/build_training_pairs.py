"""Build matched real-parent/identity drafts from training-only task records."""
import argparse,collections,hashlib,json,pathlib,sys
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from input_grounded_prompt import make_prompt
from schema_gate import parse_calls,structured_refine_gate
p=argparse.ArgumentParser();p.add_argument('--tasks',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);a=p.parse_args()
def read(path):return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]
rows=read(a.tasks);predictions=read(a.predictions)
if len({r['id'] for r in rows})!=len(rows) or len({r['id'] for r in predictions})!=len(predictions):raise ValueError('Duplicate task/prediction IDs')
pred={x['id']:x for x in predictions}
if set(pred)!={r['id'] for r in rows}:raise ValueError('Training task/prediction coverage mismatch')
out=pathlib.Path(a.output)
if out.exists() and any(out.iterdir()):raise FileExistsError('Use an empty output directory')
out.mkdir(parents=True,exist_ok=True);pairs={'identity':[],'repair':[]};counts=collections.Counter()
for row in rows:
 if row.get('split') not in (None,'train'):raise ValueError('Only training rows may select the repair cohort')
 try:
  gold=parse_calls(row['targets']);draft=parse_calls(pred[row['id']]['output'])
  if not gold or collections.Counter(c['name'] for c in gold)!=collections.Counter(c['name'] for c in draft):counts['no_call_or_name_mismatch']+=1;continue
  selected,audit=structured_refine_gate(row['inputs'],pred[row['id']]['output'],row['targets'])
  if not audit['accepted']:counts['gate_rejected']+=1;continue
 except (ValueError,KeyError,TypeError):counts['parse_error']+=1;continue
 target=''.join('<tool_call>\n'+json.dumps(c,ensure_ascii=False)+'\n</tool_call>\n' for c in gold)+'<|im_end|>'
 for arm,d in [('identity',gold),('repair',draft)]:pairs[arm].append({'id':row['id'],'inputs':make_prompt(row['inputs'],d),'targets':target})
 counts['eligible']+=1
if not pairs['repair']:raise ValueError('No eligible paired training examples')
hashes={}
for arm,records in pairs.items():
 path=out/f'{arm}.jsonl';path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records));hashes[arm]=hashlib.sha256(path.read_bytes()).hexdigest()
(out/'manifest.json').write_text(json.dumps({'rows':len(pairs['repair']),'counts':dict(counts),'file_sha256':hashes,'split':'train','historical_training_rows':590,'historical_cohort_recreated':False,'selection':'same-name/count parent draft; frozen input/schema gate accepts gold; compare with historical manifest separately'},indent=2)+'\n')
print(json.dumps(dict(counts),indent=2))
