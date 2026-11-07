"""CPU-only report replay and integrity validation; never launches GPU work."""
import argparse,hashlib,json,pathlib
R=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
def load(n):return json.loads((R/'results'/n).read_text())
if (R/'results/toolace2500.json').exists():
 d=load('toolace2500.json');assert d['rows']==2500 and d['call']==2000 and d['no_call']==500
 assert set(d['per_seed_all_mandatory'])=={'42','43','44'}
 means=d['three_seed_mean_accuracy'];print('ToolACE2500 | call exact | no-call | macro')
 for k,v in means.items():print(f"{k:10s} | {100*v['call']:.2f}% | {100*v['no_call']:.2f}% | {100*v['macro']:.2f}%")
 assert abs(sum(d['per_seed_all_mandatory'][s]['repair']['tool_exact_match'] for s in ['42','43','44'])/3-means['repair']['call'])<1e-12
 assert load('postflight-audit.json')['overall_pass'] is True
 print('Scope: reference-exact; not executable tool success. No statistical superiority over posthoc static control established.')
elif (R/'results/mainline.json').exists():
 d=load('mainline.json');assert d['stage']=='complete' and d['prediction_coverage_audited'] and d['all_stored_scores_recomputed'];assert len(d['arms'])==8
 for ds in ['toolace','glaive']:
  for arm in ['base','sft','grpo','sequential']:
   row=d['arms'][f'{ds}/{arm}'];assert set(row)=={'val','test'};assert row['test']['conversations']==1000
   print(f"{ds:8s}/{arm:10s}: {100*row['test']['exact_match']:.2f}% (n={row['test']['assistant_targets']} assistant targets)")
 print('Scope: given reference histories, normalized exact match; not live environment success.')
else:
 d=load('benchmarks.json');assert d['api_bank']['samples']==597 and d['bfcl']['samples_per_model']==4441 and d['bfcl']['official_checker_completed']
 print('API-Bank:',round(d['api_bank']['raw_accuracy']*100,2),'→',round(d['api_bank']['grpo_accuracy']*100,2))
 print('BFCL V3 overall:',d['bfcl']['metrics']['Overall Acc'])
 assert d['bfcl']['infrastructure_errors']==0
 print('Scope: two completed benchmark suites; Bamboogle not evaluated; published headline not fully reproduced.')
if a.check:
 manifest=R/'results/publication-manifest.json'
 if manifest.exists():
  for name,digest in json.loads(manifest.read_text())['sha256'].items():
   assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,f'Changed publication artifact: {name}'
 print('Artifact checks passed. Historical GPU experiments were not rerun.')
