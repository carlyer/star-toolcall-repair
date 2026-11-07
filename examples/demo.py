"""Synthetic CPU example: preserve an explicit email and reject unsupported edits."""
import json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from schema_gate import structured_refine_gate
tool={'name':'check_email','parameters':{'type':'object','properties':{'email':{'type':'string'}},'required':['email']}}
prompt='<tools>\n'+json.dumps(tool)+'\n</tools>\n<|im_start|>user\nCheck the email alex@example.test exactly as written.<|im_end|>'
def call(email):return '</think>\n<tool_call>'+json.dumps({'name':'check_email','arguments':{'email':email}})+'</tool_call>'
parent=call('alex@example.test');candidate=call('alex@example.com')
chosen,audit=structured_refine_gate(prompt,parent,candidate)
print(json.dumps({'synthetic_demo':True,'audit':audit,'parent_preserved':chosen==parent},indent=2))
