from llm.client import chat
from tools.registry import (TOOLS, TOOL_MAP)
import json
import time
from agent.trace import (AgentTrace, StepTrace, ToolTrace)

TOOL_SCHEMAS = [
  tool.schema() for tool in TOOLS
]

SYSTEM_PROMPT = """
You are a GitHub repository analysis agent.

Use the available tools to inspect repositories and answer questions
using evidence from actual repository files.

Rules:

1. Do not guess repository details.
2. Clone a repository when needed.
3. Use summarize_repository to obtain a high-level overview.
4. Inspect the repository structure before selecting files.
5. Read the README when it exists.
6. Read relevant source and configuration files before explaining code.
7. Use search_code when you do not know where something is implemented.
8. Avoid reading every file.
9. If a tool fails, inspect the error and try a reasonable alternative.
10. Only give the final answer after gathering enough evidence.
"""

class Agent:
  def __init__(self):
    self.tools = TOOL_MAP
  
  def _preview(self, value, max_chars: int = 500) -> str:
    try:
      text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
      text = str(value)

    if len(text) <= max_chars:
      return text
    
    return text[:max_chars] + "...[truncated]"

  def run(self, user_input, max_steps = 10) -> str:
    self.trace = AgentTrace()

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

    for step_number in range(1, max_steps + 1):
      print(f"\n===== Step {step_number} =====")

      step_trace = StepTrace(step=step_number)

      response = chat(messages, TOOL_SCHEMAS)

      if not response.tool_calls:
        final_response = response.content or ""
        step_trace.final_response = final_response

        self.trace.add_step(step_trace)
        self.trace.finish("completed")

        return final_response

      messages.append(response.model_dump(exclude_none=True))

      for tool_call in response.tool_calls:
        result, tool_trace = self.execute_tool(tool_call)

        step_trace.tool_calls.append(tool_trace)

        status = "success" if tool_trace.success else "failed"

        print(
          f"Tool: {tool_trace.tool_name} "
          f"[{status}] "
          f"{tool_trace.duration_ms:.2f} ms"
        )

        content = json.dumps(
          result,
          ensure_ascii=False,
          default=str,
        ) #原先的str(result)得到的字符串里的引号(python格式)是单引号，而json是双引号；某些复杂对象无法表达。

        messages.append({
          "role": "tool",
          "tool_call_id": tool_call.id,
          "content": content
        })
        
      self.trace.add_step(step_trace)
      
    self.trace.finish("max_steps_reached")

    return "The agent reached the maximum number of steps before producing a final answer."
      
  def execute_tool(self, tool_call)-> tuple[dict, ToolTrace]:
    name = tool_call.function.name
    started_at = time.perf_counter()

    if name not in self.tools:
      result = {
          "success": False,
          "error": f"Tool does not exist: {name}",
      }

      duration_ms = (
        time.perf_counter() - started_at) * 1000

      trace = ToolTrace(
        tool_call_id=tool_call.id,
        tool_name=name,
        arguments=arguments,
        success=False,
        duration_ms=duration_ms,
        result_preview=self._preview(result),
        error=result["error"],
      )

      return result, trace
    
    try:
      arguments = json.loads(tool_call.function.arguments)

    except json.JSONDecodeError as error:
      result = {
        "success": False,
        "error": f"Invalid JSON arguments: {error}"
      }

      duration_ms = (time.perf_counter() - started_at) * 1000

      trace = ToolTrace(
        tool_call_id = tool_call.id,
        tool_name = name,
        arguments = {},
        success = False,
        duration_ms = duration_ms,
        result_preview = self._preview(result),
        error = result["error"]
      )


      return result, trace
    
    tool = self.tools[name]

    try:
      result = tool.execute(**arguments)

    except TypeError as error:
      result = {
        "success": False,
        "error": f"Invalid tool parameters: {error}",
      }
    
    except Exception as error:
      result = {
        "success": False,
        "error": f"Tool execution failed: {error}",
      }
    
    duration_ms = (
        time.perf_counter() - started_at
    ) * 1000

    success = (
        isinstance(result, dict)
        and result.get("success", True)
    )

    error_message = None

    if not success and isinstance(result, dict):
        error_message = result.get("error")

    trace = ToolTrace(
        tool_call_id=tool_call.id,
        tool_name=name,
        arguments=arguments,
        success=bool(success),
        duration_ms=duration_ms,
        result_preview=self._preview(result),
        error=error_message,
    )

    return result, trace