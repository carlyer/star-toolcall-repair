"""Public STAR components: preserve Hammer split, add released positive examples."""
import hashlib
import json
import random
import os
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/generic-teacher-public-mixed'
DATA.mkdir(exist_ok=True)
TOKENIZER = AutoTokenizer.from_pretrained(os.environ.get('STUDENT_MODEL','Qwen/Qwen3-0.6B'))
positives = [json.loads(line) for line in (ROOT/'upstream/example_messages.jsonl').open()]
key = lambda row: hashlib.sha256(json.dumps({'messages':row['messages'][:-1], 'tools':row['tools']}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
ordered = sorted(positives, key=key)
random.Random(42).shuffle(ordered)
assert len({key(row) for row in ordered}) == len(ordered) == 1024
partitions = {'train':ordered[:819], 'val':ordered[819:921], 'test':ordered[921:]}
counts = {}
prompts = {}
for split, positive in partitions.items():
    negative = [json.loads(line) for line in (ROOT/f'data/generic-teacher-hammer/{split}_messages.jsonl').open()]
    raw = negative+positive
    for filename, rows in [(f'{split}_messages.jsonl', raw), (f'{split}_positive_messages.jsonl', positive)]:
        with (DATA/filename).open('w') as handle:
            for row in rows: handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    transformed = []
    lengths = []
    for row in raw:
        prompt = TOKENIZER.apply_chat_template(row['messages'][:-1], tools=row['tools'], tokenize=False, add_generation_prompt=True)+'<think>'
        target = TOKENIZER.apply_chat_template(row['messages'][-1:], tokenize=False).replace('<|im_start|>assistant\n', '')
        if target.startswith('<think>'): target = target[7:]
        if '</think>' not in target: target = '</think>'+target
        transformed.append({'id':row['id'], 'inputs':prompt, 'targets':target})
        lengths.append(len(TOKENIZER.encode(prompt, add_special_tokens=False)))
    assert max(lengths) <= 5120
    with (DATA/f'{split}.jsonl').open('w') as handle:
        for row in transformed: handle.write(json.dumps(row, ensure_ascii=False)+'\n')
    prompts[split] = {row['inputs'] for row in transformed}
    counts[split] = {'rows':len(raw), 'positive_tool_call_rows':len(positive), 'no_tool_rows':len(negative), 'max_prompt_tokens':max(lengths), 'unique_model_prompts':len(prompts[split])}
assert not prompts['train'] & prompts['val']
assert not prompts['train'] & prompts['test']
assert not prompts['val'] & prompts['test']
manifest = {'experiment':'generic-teacher-public-mixed', 'seed':42, 'split_by':'prompt groups; existing negative split preserved', 'components':{'positive':'Official STAR released example_messages.jsonl, 1024 XLAM examples; not full gated XLAM', 'negative':'All 6763 official Hammer irrelevance rows'}, 'splits':counts, 'model_prompt_overlap_between_splits':0, 'original_full_paper_mixture':False, 'mixture_reweighted':False}
(DATA/'split_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest, indent=2))
