"""Deterministic prompt-group split; all public Hammer rows assigned once."""
import collections, hashlib, json, random
from pathlib import Path
root = Path(__file__).resolve().parents[1]
data=root/'data/generic-teacher-hammer'
data.mkdir(exist_ok=True)
rows=[json.loads(l) for l in (root/'upstream/data/hammer_messages.jsonl').open()]
groups=collections.defaultdict(list)
for row in rows:
    prompt={'messages':row['messages'][:-1], 'tools':row['tools']}
    key=hashlib.sha256(json.dumps(prompt,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    groups[key].append(row)
keys=sorted(groups)
random.Random(42).shuffle(keys)
a,b=int(len(keys)*.8),int(len(keys)*.9)
counts={}
for name,subset in [('train',keys[:a]),('val',keys[a:b]),('test',keys[b:])]:
    flat=[row for key in subset for row in groups[key]]
    with (data/f'{name}_messages.jsonl').open('w') as h:
        for row in flat:h.write(json.dumps(row,ensure_ascii=False)+'\n')
    counts[name]={'rows':len(flat),'prompt_groups':len(subset),'prompt_hashes':subset}
manifest={'experiment':'generic-teacher-hammer','seed':42,'teacher':'models/Qwen/Qwen3-8B','student':'models/Qwen/Qwen3-0.6B','original_paper_teacher':False,'original_paper_full_data':False,'total_rows':len(rows),'splits':counts,'group_by':'sha256(messages_except_last + tools), sorted then seeded shuffle'}
(data/'split_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print({k:v['rows'] for k,v in counts.items()})
