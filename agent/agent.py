from llm.client import chat
from tools.registry import TOOLS
from tools.registry import TOOL_MAP
import json

tool_schemas = [
  tool.schema() for tool in TOOLS
]

SYSTEM_PROMPT = """
You are a GitHub repository analysis agent.

Use the available tools to inspect repositories and answer questions
using evidence from actual repository files.

Rules:
1. Do not guess repository details.
2. Clone a repository when needed.
3. Inspect the repository structure before selecting files.
4. Read the README when it exists.
5. Read relevant source and configuration files before explaining code.
6. Use search_code when you do not know where something is implemented.
7. Avoid reading every file.
8. If a tool fails, inspect the error and try a reasonable alternative.
9. Only give the final answer after gathering enough evidence.
"""

class Agent:
  def __init__(self):
    self.tools = TOOL_MAP
  
  def run(self, user_input, max_steps = 10):
    messages = [
      {
        "role": "system",
        "content": SYSTEM_PROMPT
      },
      {
        "role": "user",
        "content": user_input
      }
    ]

    for step in range(1, max_steps + 1):
      print(f"\n===== Step {step} =====")

      response = chat(messages, tool_schemas)

      if response.tool_calls:
        messages.append(response.model_dump())

        for tool_call in response.tool_calls:
          result = self.execute_tool(tool_call)

          messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
          })

      else:
        return response.content
      
    return "Maximum steps reached"
      
  def execute_tool(self, tool_call):
    name = tool_call.function.name

    if name not in self.tools:
      print(f"call tool: {name} failed, {name} doesn't exist.")
      return f"Tool {name} does not exist"

    arguments = json.loads(tool_call.function.arguments)

    tool = self.tools[name]
    result = tool.execute(**arguments)

    print(f"call tool: {name}") #
    print(f"{result}")

    return result