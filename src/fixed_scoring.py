"""The same STAR primary scoring rules, separate from generation and gating."""
import pathlib,importlib.util,json,collections,hashlib
R=pathlib.Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('star_sim_reward',R/'upstream/simrl.py');rules=importlib.util.module_from_spec(spec);spec.loader.exec_module(rules)
def score(row,text,finish):
 _,gold,reply_gold=rules.get_ground_truth_from_label(row['targets']);calls=[];valid=False;exact=False;similarity=0.;attempt='<tool_call' in text.split('</think>')[-1]
 try:
  _,calls,reply=rules.parse_generation('<think>'+text);valid=rules.check_tool_calls_valid(calls,rules.extract_tools_from_prompt(row['inputs']))
  if attempt and not calls:valid=False
  if valid:
   if gold:
    canonical=lambda c:collections.Counter(json.dumps(x,sort_keys=True,ensure_ascii=False) for x in c)
    exact=canonical(gold)==canonical(calls);similarity=rules.compute_function_calling_reward(gold,calls)
   elif reply:
    clean=reply_gold.replace('<|im_end|>','').strip();exact=reply.strip()==clean;similarity=rules.get_rouge_score(reply,clean)
 except Exception:pass
 return {'id':row['id'],'prompt_sha256':hashlib.sha256(row['inputs'].encode()).hexdigest(),'format_valid':bool(valid),'tool_case':bool(gold),'predicted_tool_call':bool(calls) or attempt,'exact':bool(exact),'similarity':float(similarity),'output':text,'finish_reason':finish}
def summary(rows):
 pos=[x for x in rows if x['tool_case']];neg=[x for x in rows if not x['tool_case']];tp=sum(x['exact'] for x in pos)/len(pos) if pos else None;tn=sum(x['format_valid'] and not x['predicted_tool_call'] for x in neg)/len(neg) if neg else None
 return {'rows':len(rows),'tool_rows':len(pos),'no_tool_rows':len(neg),'tool_exact_match':tp,'no_tool_decision_accuracy':tn,'macro_policy_accuracy':(tp+tn)/2 if tp is not None and tn is not None else None,'overall_policy_accuracy':sum(x['exact'] if x['tool_case'] else x['format_valid'] and not x['predicted_tool_call'] for x in rows)/len(rows)}
