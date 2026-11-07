"""Independent fixed-test scoring of positive calls and no-call decisions."""
import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
from vllm import LLM, SamplingParams

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='generic-teacher-hammer')
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tensor-parallel-size', type=int, default=2)
    parser.add_argument('--adapter')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('simrl', ROOT/'upstream/simrl.py')
    reward = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reward)
    rows = [json.loads(line) for line in Path(args.data).open()]
    llm = LLM(model=args.model, dtype='bfloat16', tensor_parallel_size=args.tensor_parallel_size, enable_lora=bool(args.adapter), max_lora_rank=16,
              gpu_memory_utilization=.65, max_model_len=5632, enforce_eager=True)
    outputs = llm.generate([row['inputs'] for row in rows],
                           SamplingParams(temperature=0, max_tokens=512),
                           lora_request=__import__('vllm.lora.request', fromlist=['LoRARequest']).LoRARequest('calibration', 1, args.adapter) if args.adapter else None)
    canonical = lambda calls: collections.Counter(
        json.dumps(call, sort_keys=True, ensure_ascii=False) for call in calls)
    records = []
    for row, output in zip(rows, outputs):
        text = output.outputs[0].text
        _, gold_calls, gold_reply = reward.get_ground_truth_from_label(row['targets'])
        calls, valid, exact, similarity = [], False, False, 0.0
        attempted_call = '<tool_call' in text.split('</think>')[-1]
        try:
            _, calls, reply = reward.parse_generation('<think>'+text)
            valid = reward.check_tool_calls_valid(calls, reward.extract_tools_from_prompt(row['inputs']))
            if attempted_call and not calls:
                valid = False
            if valid:
                if gold_calls:
                    exact = canonical(gold_calls) == canonical(calls)
                    similarity = reward.compute_function_calling_reward(gold_calls, calls)
                elif reply:
                    clean_gold = gold_reply.replace('<|im_end|>', '').strip()
                    similarity = reward.get_rouge_score(reply, clean_gold)
                    exact = reply.strip() == clean_gold
        except Exception:
            pass
        records.append({
            'id': row['id'], 'prompt_sha256': hashlib.sha256(row['inputs'].encode()).hexdigest(),
            'format_valid': bool(valid), 'tool_case': bool(gold_calls),
            'predicted_tool_call': bool(calls) or attempted_call, 'exact': bool(exact),
            'similarity': float(similarity), 'output': text,
            'finish_reason': output.outputs[0].finish_reason,
        })
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix('.predictions.jsonl').open('w') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False)+'\n')
    tools = [record for record in records if record['tool_case']]
    no_tools = [record for record in records if not record['tool_case']]
    tool_exact = sum(record['exact'] for record in tools)/len(tools) if tools else None
    no_tool_accuracy = sum(record['format_valid'] and not record['predicted_tool_call']
                           for record in no_tools)/len(no_tools) if no_tools else None
    report = {
        'experiment': args.experiment, 'model': args.model, 'adapter':args.adapter, 'data': args.data,
        'rows': len(records), 'unique_prompt_groups': len({record['prompt_sha256'] for record in records}),
        'tool_rows': len(tools), 'gold_no_tool_rows': len(no_tools),
        'tool_exact_match': tool_exact, 'no_tool_decision_accuracy': no_tool_accuracy,
        'macro_policy_accuracy': (tool_exact+no_tool_accuracy)/2 if tools and no_tools else None,
        'mean_tool_similarity': sum(record['similarity'] for record in tools)/len(tools) if tools else None,
        'mean_reply_rougeL': sum(record['similarity'] for record in no_tools)/len(no_tools) if no_tools else None,
        'overall_policy_accuracy': sum(record['exact'] if record['tool_case'] else (record['format_valid'] and not record['predicted_tool_call']) for record in records)/len(records),
        'format_valid_rate': sum(record['format_valid'] for record in records)/len(records),
        'mean_similarity': sum(record['similarity'] for record in records)/len(records),
        'generation_length_finish_rate': sum(record['finish_reason']=='length' for record in records)/len(records),
        'max_generation_tokens': 512, 'temperature': 0,
        'reply_reference_chat_eos_stripped': True,
        'independent_prompt_split': True, 'original_paper_benchmark': False,
        'dataset_component': 'Frozen synthetic ToolMind transfer subset' if 'blind' in args.experiment else ('Hammer irrelevance only' if not tools else 'Released XLAM positive subset + Hammer irrelevance'),
    }
    destination.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
