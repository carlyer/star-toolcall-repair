"""Parent-decision-preserving schema gate. No labels or expected arguments are inputs."""
import json,re,collections,copy,datetime,calendar
import jsonschema
ALIASES={'str':'string','int':'integer','float':'number','bool':'boolean','dict':'object','list':'array','tuple':'array'}
def normalize_schema(value):
 value=copy.deepcopy(value)
 if not isinstance(value,dict):raise ValueError('Unsupported schema')
 typ=value.get('type')
 if isinstance(typ,str):
  typ=typ.split(',')[0].strip();value['type']=ALIASES.get(typ,typ)
 if value.get('type')=='array' and 'items' not in value:value['items']={}
 if isinstance(value.get('properties'),dict):value['properties']={k:normalize_schema(v) for k,v in value['properties'].items()}
 if isinstance(value.get('items'),dict):value['items']=normalize_schema(value['items'])
 return value

def schemas_from_prompt(prompt):
 blocks=re.findall(r'<tools>(.*?)</tools>',prompt,re.S)
 if not blocks:raise ValueError('No tool definitions')
 schemas={}
 for line in blocks[-1].splitlines():
  if not line.strip():continue
  tool=json.loads(line);p=tool['parameters']
  if not isinstance(p,dict):raise ValueError('Unsupported parameters')
  if 'properties' not in p and not isinstance(p.get('type'),(str,list)):
   p={'type':'object','properties':p,'required':[k for k,v in p.items() if isinstance(v,dict) and v.get('required') is True]}
  try:
   p=normalize_schema(p);jsonschema.Draft7Validator.check_schema(p)
  except (ValueError,TypeError,jsonschema.exceptions.SchemaError):continue
  schemas[tool['name']]=p
 return schemas

def parse_calls(text):
 # Generated continuation begins inside <think>; require one completed thought block.
 full='<think>'+text
 if full.count('<think>')!=1 or full.count('</think>')!=1 or len(re.findall(r'<think>(.*?)</think>',full,re.S))!=1:raise ValueError('Malformed reasoning delimiter')
 answer=full.split('</think>')[-1]
 blocks=re.findall(r'<tool_call>(.*?)</tool_call>',answer,re.S)
 if answer.count('<tool_call')!=len(blocks) or answer.count('</tool_call>')!=len(blocks):raise ValueError('Malformed tool call')
 calls=[json.loads(b) for b in blocks]
 for c in calls:
  if not isinstance(c,dict) or not isinstance(c.get('name'),str) or not isinstance(c.get('arguments'),dict):raise ValueError('Malformed call object')
 return calls

def violations(calls,schemas):
 problems=[]
 for c in calls:
  if c['name'] not in schemas:problems.append('unknown_tool');continue
  schema=schemas[c['name']]
  if isinstance(schema.get('properties'),dict) and not set(c['arguments']).issubset(schema['properties']):problems.append('undeclared_argument')
  problems.extend(e.validator for e in jsonschema.Draft7Validator(schema).iter_errors(c['arguments']))
 return sorted(set(problems))

def gate(prompt,parent_output,adapter_output):
 audit={'triggered':False,'accepted':False,'reason':None,'parent_violations':[],'adapter_violations':[]}
 try:schemas=schemas_from_prompt(prompt);parent=parse_calls(parent_output)
 except (ValueError,KeyError,TypeError,jsonschema.exceptions.SchemaError) as exc:audit['reason']='unsupported_parent_or_schema';return parent_output,audit
 if not parent:audit['reason']='parent_no_call_preserved';return parent_output,audit
 errors=violations(parent,schemas);audit['parent_violations']=errors
 if not errors:audit['reason']='parent_schema_valid_preserved';return parent_output,audit
 audit['triggered']=True
 try:child=parse_calls(adapter_output)
 except (ValueError,KeyError,TypeError):audit['reason']='adapter_parse_failure';return parent_output,audit
 if collections.Counter(c['name'] for c in child)!=collections.Counter(c['name'] for c in parent):audit['reason']='tool_name_or_count_changed';return parent_output,audit
 errors=violations(child,schemas);audit['adapter_violations']=errors
 if errors:audit['reason']='adapter_schema_invalid';return parent_output,audit
 audit.update(accepted=True,reason='schema_repair_accepted');return adapter_output,audit


def refine_gate(prompt,parent_output,refined_output):
 """Accept only a valid same-name/count refinement; preserve all no-call decisions."""
 audit={'triggered':False,'accepted':False,'reason':None}
 try:parent=parse_calls(parent_output)
 except (ValueError,KeyError,TypeError):audit['reason']='parent_parse_failure';return parent_output,audit
 if not parent:audit['reason']='parent_no_call_preserved';return parent_output,audit
 audit['triggered']=True
 try:schemas=schemas_from_prompt(prompt);child=parse_calls(refined_output)
 except (ValueError,KeyError,TypeError,jsonschema.exceptions.SchemaError):audit['reason']='unsupported_schema_or_refinement';return parent_output,audit
 if collections.Counter(c['name'] for c in child)!=collections.Counter(c['name'] for c in parent):audit['reason']='name_or_count_changed';return parent_output,audit
 problems=violations(child,schemas);audit['refinement_violations']=problems
 if problems:audit['reason']='schema_invalid_refinement';return parent_output,audit
 audit.update(accepted=True,reason='accepted_same_decision_refinement');return refined_output,audit

def user_text(prompt):
 return '\n'.join(re.findall(r'<\|im_start\|>user\n(.*?)<\|im_end\|>',prompt,re.S))

def anchored(value,text):
 if isinstance(value,bool):return bool(re.search(r'(?<!\w)'+str(value).lower()+r'(?!\w)',text.lower()))
 if isinstance(value,(int,float)) or isinstance(value,str) and re.fullmatch(r'-?\d+(\.\d+)?',value):return bool(re.search(r'(?<![\w.])'+re.escape(str(value))+r'(?!\w|\.\d)',text))
 if isinstance(value,str):return bool(value) and value in text
 return False

def same_literal(a,b):
 if type(a)==type(b):return a==b
 if isinstance(a,(str,int,float)) and not isinstance(a,bool) and isinstance(b,(str,int,float)) and not isinstance(b,bool):return str(a)==str(b)
 return False

def leaves(value,path=()):
 if isinstance(value,dict):
  for k,v in value.items():yield from leaves(v,path+(k,))
 elif isinstance(value,list):
  for k,v in enumerate(value):yield from leaves(v,path+(k,))
 else:yield path,value

def at_path(value,path):
 for k in path:value=value[k]
 return value

def explicit_date_anchor(value,text,schema,path):
 """Only ISO date values with tool-field evidence and an explicit named user date."""
 if not isinstance(value,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):return False
 field=schema
 for key in path:
  field=field.get('items',{}) if isinstance(key,int) else field.get('properties',{}).get(key,{})
 evidence=str(path[-1] if path else '')+' '+str(field.get('description',''))
 if field.get('format')!='date' and not re.search(r'date|birth|dob|yyyy',evidence,re.I):return False
 try:wanted=datetime.date.fromisoformat(value)
 except ValueError:return False
 months={name.lower():i for i,name in enumerate(calendar.month_name) if name}
 pattern=r'(?<!\w)('+ '|'.join(months) +r')\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(\d{4})(?!\d)'
 for m in re.finditer(pattern,text,re.I):
  try:found=datetime.date(int(m[3]),months[m[1].lower()],int(m[2]))
  except ValueError:continue
  if found==wanted:return True
 return False

def grounding_problems(prompt,parent,child):
 text=user_text(prompt);schemas=schemas_from_prompt(prompt);problems=[]
 buckets=collections.defaultdict(list)
 for c in parent:buckets[c['name']].append(c)
 for c in child:
  old=buckets[c['name']].pop(0);required=set(schemas.get(c['name'],{}).get('required',[]))
  for path,value in leaves(old['arguments']):
   if not anchored(value,text) and not explicit_date_anchor(value,text,schemas.get(c['name'],{}),path):continue
   try:new=at_path(c['arguments'],path)
   except (KeyError,IndexError,TypeError):problems.append('removed_explicit_user_literal');continue
   if not same_literal(value,new):problems.append('changed_explicit_user_literal')
  for key,value in c['arguments'].items():
   values=list(leaves(value))
   supported=bool(values) and all(anchored(v,text) for _,v in values)
   if key not in old['arguments'] and key not in required and not supported:problems.append('unanchored_optional_addition')
 return sorted(set(problems))

def grounded_refine_gate(prompt,parent_output,refined_output):
 selected,audit=refine_gate(prompt,parent_output,refined_output)
 if not audit['accepted']:return selected,audit
 problems=grounding_problems(prompt,parse_calls(parent_output),parse_calls(refined_output));audit['grounding_violations']=problems
 if problems:audit.update(accepted=False,reason='grounding_rejected');return parent_output,audit
 return selected,audit

"""Development-only structured preservation proposal; no target arguments consumed."""
import re,collections,datetime
WORDS={'one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7','eight':'8','nine':'9','ten':'10','eleven':'11','twelve':'12'}
def field_schema(schema,path):
 for k in path:schema=schema.get('items',{}) if isinstance(k,int) else schema.get('properties',{}).get(k,{})
 return schema
def duration_value(value):
 m=re.fullmatch(r'(\d+(?:\.\d+)?)\s*(years?|yrs?|Y|months?|mo|M|weeks?|W|days?|D|hours?|hrs?|H|minutes?|mins?|seconds?|secs?|S)',value)
 if not m:return None
 unit=m[2];aliases={'Y':'year','M':'month','W':'week','D':'day','H':'hour','S':'second','yr':'year','yrs':'year','mo':'month','hr':'hour','hrs':'hour','min':'minute','mins':'minute','sec':'second','secs':'second'}
 return (float(m[1]),aliases.get(unit,unit.rstrip('s')))
TIME_SCALE={'week':604800,'day':86400,'hour':3600,'minute':60,'second':1}
def explicit_numeric_duration(value,desc,source):
 if isinstance(value,bool) or not isinstance(value,(int,float)):return False
 target_units={m.rstrip('s') for m in re.findall(r'\b(?:weeks?|days?|hours?|minutes?|seconds?)\b',desc)}
 if len(target_units)!=1:return False
 target=next(iter(target_units));source=re.sub(r'\bhalf (?:a|an) (week|day|hour|minute|second)\b',lambda m:'0.5 '+m[1],source)
 source=re.sub(r'\b(?:a|an) (week|day|hour|minute|second)\b',lambda m:'1 '+m[1],source)
 for m in re.finditer(r'(?<!\w)(\d+(?:\.\d+)?)\s+(weeks?|days?|hours?|minutes?|seconds?)(?!\w)',source):
  converted=float(m[1])*TIME_SCALE[m[2].rstrip('s')]/TIME_SCALE[target]
  if abs(float(value)-converted)<1e-9:return True
 return False

def problems(prompt,parent,child):
 source=user_text(prompt).lower();source=re.sub(r'\b('+ '|'.join(WORDS)+r')\b',lambda m:WORDS[m[0]],source);schemas=schemas_from_prompt(prompt);buckets=collections.defaultdict(list);errors=[]
 for c in parent:buckets[c['name']].append(c)
 for c in child:
  old=buckets[c['name']].pop(0)
  for path,a in leaves(old['arguments']):
   try:b=at_path(c['arguments'],path)
   except (KeyError,IndexError,TypeError):continue
   definition=field_schema(schemas.get(c['name'],{}),path);desc=definition.get('description','').lower()
   date_evidence=definition.get('format')=='date' or re.search(r'date|birth|dob|yyyy',str(path[-1] if path else '')+' '+desc,re.I)
   if date_evidence and isinstance(b,str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}',b):
    try:datetime.date.fromisoformat(b)
    except ValueError:errors.append('invalid_calendar_date')
   equivalent_number=isinstance(b,(int,float)) and not isinstance(b,bool) and a==b or isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,str) and bool(re.fullmatch(r'-?\d+(?:\.\d+)?',b)) and float(a)==float(b)
   if a!=b and not equivalent_number and explicit_numeric_duration(a,desc,source):errors.append('changed_explicit_converted_time_quantity')
   if not isinstance(a,str) or not isinstance(b,str) or a==b:continue
   if re.fullmatch(r'\d+(?:\.\d+)?\s+(?:years?|months?|weeks?|days?|hours?|minutes?|seconds?)',a.lower()) and re.search(r'(?<!\w)'+re.escape(a.lower())+r'(?!\w)',source) and duration_value(a)!=duration_value(b):errors.append('changed_explicit_word_number_duration')
   if 'city and state' in desc and re.fullmatch(r'.+,\s*[A-Z]{2}',a) and not re.fullmatch(r'.+,\s*[A-Z]{2}',b):errors.append('lost_schema_city_state_structure')
 return sorted(set(errors))

def structured_refine_gate(prompt,parent_output,refined_output):
 selected,audit=grounded_refine_gate(prompt,parent_output,refined_output)
 if not audit['accepted']:return selected,audit
 errors=problems(prompt,parse_calls(parent_output),parse_calls(refined_output));audit['structured_violations']=errors
 if errors:audit.update(accepted=False,reason='structured_preservation_rejected');return parent_output,audit
 return selected,audit
