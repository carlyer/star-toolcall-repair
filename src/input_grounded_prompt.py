"""Grounding instruction used verbatim during training and inference."""
from repair_prompt import make_prompt as base_prompt,INSTRUCTION as OLD
INSTRUCTION='Audit this unexecuted draft against the original request and tool definitions. Keep exactly the same tool names and number of calls. Fix argument errors only when the original request supplies evidence. Preserve user strings exactly: a request to check an email is not permission to correct its spelling. Preserve explicit numeric values, and do not add optional defaults without user evidence. A date may be normalized to the schema format only if its day, month and year are explicitly given. Do not invent missing facts or tool results. If the draft is already supported, copy its arguments. Output ONLY corrected <tool_call> ... </tool_call> blocks.'
def make_prompt(prompt,draft):
 return base_prompt(prompt,draft).replace(OLD,INSTRUCTION)
