"""Put an unexecuted call draft inside the original user request, not fake tool history."""
import json
INSTRUCTION='Audit the unexecuted draft below against the original user request and tool definitions. Keep exactly the same tool names and number of calls. Correct arguments only: preserve literal user strings without wrappers, use required JSON types, include required fields, and respect enums. Do not invent tool results or missing facts. Output ONLY corrected <tool_call> ... </tool_call> blocks.'
def make_prompt(original_inputs,draft):
 prefix=original_inputs.rsplit('<|im_start|>assistant\n',1)[0].rstrip('\n')
 if not prefix.endswith('<|im_end|>') or not prefix.rsplit('<|im_start|>',1)[-1].startswith('user\n'):raise ValueError('Original prompt must end in user request')
 prefix=prefix[:-len('<|im_end|>')]
 return prefix+'\n\n'+INSTRUCTION+'\nUnexecuted draft:\n'+json.dumps(draft,ensure_ascii=False)+'<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n'
