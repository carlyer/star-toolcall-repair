"""Verify synthetic trajectories remain confined to the training prompt groups."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
data=root/'data/generic-teacher-hammer'
manifest=json.loads((data/'split_manifest.json').read_text())
train=set(manifest['splits']['train']['prompt_hashes'])
heldout=set(manifest['splits']['val']['prompt_hashes'])|set(manifest['splits']['test']['prompt_hashes'])
assert not train&heldout
rows=0;groups=set();max_reasoning=0
with (data/'kd_messages.jsonl').open() as h:
    for line in h:
        row=json.loads(line)
        prompt={'messages':row['messages'][:-1],'tools':row['tools']}
        key=hashlib.sha256(json.dumps(prompt,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        assert key in train and key not in heldout
        groups.add(key);rows+=1
        max_reasoning=max(max_reasoning,len(row['messages'][-1].get('reasoning_content') or ''))
report={'synthetic_rows':rows,'covered_training_prompt_groups':len(groups),'training_prompt_groups':len(train),'heldout_leaks':0,'max_reasoning_characters':max_reasoning,'full_original_teacher':False,'teacher':'local Qwen3-8B'}
(root/'results/teacher-split-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
