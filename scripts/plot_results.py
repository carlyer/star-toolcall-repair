"""Regenerate publication-style portfolio figures from exported aggregate evidence."""
import json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=pathlib.Path(__file__).resolve().parents[1];name=R.name
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#ccd8e3','axes.labelcolor':'#35516b','text.color':'#142b43','xtick.color':'#536b80','ytick.color':'#536b80','figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'none','svg.hashsalt':'research-portfolio-v1'})
C=['#a9b9ca','#5483a6','#0e9f8c','#eeae62']
def load(n):return json.loads((R/'results'/n).read_text())
def style(ax,title):
 ax.set_title(title,loc='left',fontweight='bold',pad=18);ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True);ax.set_ylabel('Accuracy (%)')
def save(fig,name):
 fig.savefig(R/'docs/assets'/f'{name}.svg',bbox_inches='tight',metadata={'Date':None});fig.savefig(R/'docs/assets'/f'{name}.png',bbox_inches='tight',dpi=170);plt.close(fig)
if (R/'results/toolace2500.json').exists():
 d=load('toolace2500.json');means=d['three_seed_mean_accuracy'];labels=['Raw','SimRL parent','Self-reask','Identity','Repair'];keys=['raw','parent','self','identity','repair'];vals=[100*means[k]['call'] for k in keys]
 fig,ax=plt.subplots(figsize=(11.8,4.8));bars=ax.bar(labels,vals,color=[C[0],C[1],C[1],C[1],C[2]],width=.6)
 for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+.55,f'{v:.2f}',ha='center',fontweight='bold')
 style(ax,'Frozen ToolACE2500 · call reference exact');ax.set_ylim(0,50);ax.set_yticks([0,10,20,30,40,50]);ax.text(.99,.94,'Repair: +1.40 pp vs parent\n3 seeds · 2,000 unique call requests',ha='right',va='top',transform=ax.transAxes,color='#087f74',fontsize=11)
 fig.text(.08,-.005,'500 no-call requests reported separately. Reference match is not executable success.',fontsize=10,color='#637b8e');fig.tight_layout();save(fig,'results')
 p=load('public-mixed.json');fig,axes=plt.subplots(1,2,figsize=(11.8,4.5))
 for ax,key,title in zip(axes,['tool_exact_match','no_tool_decision_accuracy'],['Public mixed test · call exact (n=103)','Public mixed test · no-call (n=677)']):
  vals=[100*p['evaluations'][k][key] for k in ['base','ckd','sim-rl']];bars=ax.bar(['Base','CKD','Sim-RL'],vals,color=C[:3],width=.55)
  for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+1,f'{v:.2f}',ha='center',fontsize=10)
  style(ax,title);ax.set_ylim(0,105)
 fig.tight_layout();save(fig,'baseline')
elif (R/'results/mainline.json').exists():
 d=load('mainline.json');fig,axes=plt.subplots(1,2,figsize=(12,5));arms=['base','sft','grpo','sequential'];labels=['Base','SFT','GRPO','SFT→GRPO'];x=np.arange(4)
 for ax,ds in zip(axes,['toolace','glaive']):
  vals=[100*d['arms'][f'{ds}/{a}']['test']['exact_match'] for a in arms];bars=ax.bar(x,vals,color=C,width=.6)
  for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+1.2,f'{v:.2f}',ha='center',fontsize=10)
  ax.set_xticks(x,labels);style(ax,f'{ds.title()} · test reference exact');ax.set_ylim(0,108)
 fig.text(.09,-.005,'Qwen3-0.6B · given reference histories · SFT is best on both local test corpora.',fontsize=10,color='#637b8e');fig.tight_layout();save(fig,'results')
 fig,axes=plt.subplots(1,2,figsize=(12,5))
 for ax,ds in zip(axes,['toolace','glaive']):
  a=[100*d['arms'][f'{ds}/{m}']['test']['tool_call_exact_match'] for m in arms];b=[100*d['arms'][f'{ds}/{m}']['test']['no_call_exact_match'] for m in arms]
  ax.bar(x-.17,a,.34,label='Call targets',color=C[1]);ax.bar(x+.17,b,.34,label='No-call targets',color=C[2]);ax.set_xticks(x,labels);style(ax,f'{ds.title()} · inspect class imbalance');ax.set_ylim(0,108);ax.legend(frameon=False,fontsize=9)
 fig.tight_layout();save(fig,'class-breakdown')
else:
 d=load('benchmarks.json');fig,axes=plt.subplots(1,2,figsize=(11.8,4.8));api=d['api_bank'];bfcl=d['bfcl']['metrics']['Overall Acc'];data=[[100*api['raw_accuracy'],100*api['grpo_accuracy']],[bfcl['base_percent'],bfcl['grpo_percent']]]
 for ax,vals,title in zip(axes,data,['API-Bank · author soft match (n=597)','BFCL V3 · official overall (n=4,441)']):
  bars=ax.bar(['Base','GRPO'],vals,color=[C[0],C[2]],width=.5)
  for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+1,f'{v:.2f}',ha='center',fontweight='bold')
  style(ax,title);ax.set_ylim(0,70);ax.text(.98,.95,f'+{vals[1]-vals[0]:.2f} pp',ha='right',va='top',transform=ax.transAxes,color='#087f74',fontsize=13,fontweight='bold')
 fig.text(.09,-.005,'Each suite has its own scoring protocol; the two accuracies are not directly comparable.',fontsize=10,color='#637b8e');fig.tight_layout();save(fig,'results')
print('Figures regenerated from aggregate evidence.')
