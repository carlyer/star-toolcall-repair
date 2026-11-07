import json,pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from schema_gate import structured_refine_gate
class GateTests(unittest.TestCase):
 def setUp(self):
  tool={'name':'lookup','parameters':{'type':'object','properties':{'email':{'type':'string'},'limit':{'type':'integer'}},'required':['email']}}
  self.prompt='<tools>\n'+json.dumps(tool)+'\n</tools>\n<|im_start|>user\nLook up alex@example.test.<|im_end|>'
 def call(self,args,name='lookup'):return '</think>\n<tool_call>'+json.dumps({'name':name,'arguments':args})+'</tool_call>'
 def test_user_literal_must_survive(self):
  parent=self.call({'email':'alex@example.test'})
  selected,a=structured_refine_gate(self.prompt,parent,self.call({'email':'alex@example.com'}))
  self.assertEqual(selected,parent);self.assertFalse(a['accepted'])
 def test_no_call_is_preserved(self):
  parent='</think>\nNo suitable tool.'
  selected,a=structured_refine_gate(self.prompt,parent,self.call({'email':'alex@example.test'}))
  self.assertEqual(selected,parent);self.assertEqual(a['reason'],'parent_no_call_preserved')
 def test_tool_change_rejected(self):
  parent=self.call({'email':'alex@example.test'})
  _,a=structured_refine_gate(self.prompt,parent,self.call({'email':'alex@example.test'},'send_email'))
  self.assertEqual(a['reason'],'name_or_count_changed')
 def test_unanchored_optional_addition_rejected(self):
  parent=self.call({'email':'alex@example.test'})
  _,a=structured_refine_gate(self.prompt,parent,self.call({'email':'alex@example.test','limit':100}))
  self.assertFalse(a['accepted'])
 def test_schema_invalid_type_rejected(self):
  parent=self.call({'email':'alex@example.test'})
  _,a=structured_refine_gate(self.prompt,parent,self.call({'email':['alex@example.test']}))
  self.assertEqual(a['reason'],'schema_invalid_refinement')
if __name__=='__main__':unittest.main()
