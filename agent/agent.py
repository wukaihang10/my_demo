import json
import time
import inspect
from typing import Any

from llm.client import chat, LLMClientError
from tools.registry import TOOLS, TOOL_MAP
from agent.trace import AgentTrace, StepTrace, ToolTrace
from agent.observation import process_observation
from agent.state import RepositoryState
from agent.context import build_state_context
from agent.history import ToolHistory

# from agent.plan import build_repository_analysis_plan
from agent.planner import LLMPlanner, PlannerError

TOOL_SCHEMAS = [tool.schema() for tool in TOOLS]

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
    def __init__(self, planner: LLMPlanner | None = None):
        self.tools = TOOL_MAP
        self.trace = AgentTrace()
        self.state = RepositoryState()
        self.tool_history = ToolHistory()
        self.planner = planner or LLMPlanner()

    def _preview(self, value, max_chars: int = 500) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)

        if len(text) <= max_chars:
            return text

        return text[:max_chars] + "...[truncated]"

    def _update_state(self, tool_name: str, arguments: dict, result) -> None:
        if not isinstance(result, dict):
            return

        success = result.get("success", False)

        if not success:
            error = result.get("error", f"Tool failed: {tool_name}")

            self.state.add_error(str(error))
            return

        if tool_name == "clone_repository":
            repo_path = result.get("repo_path")

            if repo_path:
                self.state.repo_path = str(repo_path)

            self.state.phase = "understanding"

        elif tool_name == "summarize_repository":
            important_files = result.get("important_files", [])

            self.state.important_files = list(dict.fromkeys(important_files))

            summary_fields = (
                "repo_name",
                "total_files",
                "top_level_structure",
                "languages",
                "extensions",
                "readme_path",
            )

            self.state.repository_summary = {
                field: result.get(field) for field in summary_fields
            }

            self.state.phase = "reading_code"

        elif tool_name == "list_files":
            files = result.get("files", [])
            for file_path in files:
                self.state.add_list_file(file_path)

        elif tool_name == "read_file":
            file_path = result.get("file_path") or arguments.get("file_path")
            if file_path:
                self.state.add_read_file(str(file_path))

        elif tool_name == "search_code":
            keyword = result.get("keyword") or arguments.get("keyword")
            if keyword:
                self.state.add_search_keyword(str(keyword))

    def _finalize_tool_call(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        started_at: float,
    ) -> tuple[dict[str, Any], ToolTrace]:
        normalized_result = dict(result)

        success = normalized_result["success"]
        error_message: str | None = None

        if not success:
            error_message = str(
                normalized_result.get("error") or f"Tool failed: {tool_name}"
            )

            normalized_result["error"] = error_message

        duration_ms = (time.perf_counter() - started_at) * 1000

        trace = ToolTrace(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            success=success,
            duration_ms=duration_ms,
            result_preview=self._preview(normalized_result),
            error=error_message,
        )

        self._update_state(
            tool_name=tool_name,
            arguments=arguments,
            result=normalized_result,
        )

        return normalized_result, trace

    def run(
        self,
        user_input: str,
        max_steps: int = 10,
        max_tool_calls: int = 30,
        repo_url: str | None = None,
    ) -> str:
        self.trace = AgentTrace(
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
        )

        self.state = RepositoryState(repo_url=repo_url)

        self.tool_history = ToolHistory()

        if max_steps <= 0:
            error_message = "max_steps must be greater than zero."

            self.state.add_error(error_message)
            self.state.phase = "failed"

            if self.state.plan is not None:
                self.state.plan.fail(error_message)

            self.trace.finish("invalid_max_steps")

            return error_message

        if max_tool_calls <= 0:
            error_message = "max_tool_calls must be greater than zero."

            self.state.add_error(error_message)
            self.state.phase = "failed"

            if self.state.plan is not None:
                self.state.plan.fail(error_message)

            self.trace.finish("invalid_max_tool_calls")
            return error_message

        try:
            plan = self.planner.create_plan(user_input)
            plan.start()
        except PlannerError as error:
            error_message = f"Agent planning failed: {error}"

            self.state.add_error(error_message)
            self.state.phase = "failed"
            self.trace.finish("planning_error")

            return error_message

        self.state.plan = plan

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        for step_number in range(1, max_steps + 1):
            print(f"\n===== Step {step_number} =====")

            step_trace = StepTrace(step=step_number)

            state_context = build_state_context(self.state)

            request_messages = [*messages, {"role": "system", "content": state_context}]
            try:
                response = chat(request_messages, TOOL_SCHEMAS)
            except LLMClientError as error:
                error_message = f"LLM request failed at step {step_number}: {error}"

                step_trace.error = error_message

                self.state.add_error(error_message)
                self.state.phase = "failed"

                if self.state.plan is not None:
                    self.state.plan.fail(error_message)

                self.trace.add_step(step_trace)
                self.trace.finish("llm_error")

                return "The language model request failed before the agent could complete the task. Check the repository state and agent trace for details."

            if not response.tool_calls:
                final_response = response.content or ""
                step_trace.final_response = final_response

                self.trace.add_step(step_trace)
                self.trace.finish("completed")

                self.state.phase = "completed"

                if self.state.plan is not None:
                    self.state.plan.finish(result=final_response)

                return final_response

            # 下面是工具调用循环
            messages.append(response.model_dump(exclude_none=True))

            for tool_call in response.tool_calls:
                # 检查工具调用次数是否超了

                if self.trace.tool_calls_used >= max_tool_calls:
                    error_message = "The agent reached the maximum number of allowed tool calls before producing a final answer."

                    self.trace.add_step(step_trace)
                    self.trace.finish("tool_budget_exceeded")

                    self.state.add_error(error_message)
                    self.state.phase = "failed"

                    if self.state.plan is not None:
                        self.state.plan.fail(error_message)

                    return error_message

                self.trace.record_tool_call()

                result, tool_trace = self.execute_tool(tool_call)

                step_trace.tool_calls.append(tool_trace)

                status = "success" if tool_trace.success else "failed"

                print(
                    f"Tool: {tool_trace.tool_name} "
                    f"[{status}] "
                    f"{tool_trace.duration_ms:.2f} ms"
                )

                content = process_observation(result)

                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": content}
                )

            self.trace.add_step(step_trace)

        error_message = "The agent reached the maximum number of steps before producing a final answer."

        # self.trace.add_step(step_trace)
        self.trace.finish("max_steps_reached")

        self.state.add_error(error_message)
        self.state.phase = "failed"

        if self.state.plan is not None:
            self.state.plan.fail(error_message)

        return error_message

    def execute_tool(self, tool_call) -> tuple[dict, ToolTrace]:
        name = tool_call.function.name
        started_at = time.perf_counter()

        try:
            arguments = json.loads(tool_call.function.arguments)

        except (json.JSONDecodeError, TypeError) as error:
            result = {"success": False, "error": f"Invalid JSON arguments: {error}"}

            return self._finalize_tool_call(
                tool_call_id=tool_call.id,
                tool_name=name,
                arguments={},
                result=result,
                started_at=started_at,
            )

        if not isinstance(arguments, dict):
            result = {
                "success": False,
                "error": "Tool arguments must be a JSON object",
            }

            return self._finalize_tool_call(
                tool_call_id=tool_call.id,
                tool_name=name,
                arguments={},
                result=result,
                started_at=started_at,
            )

        if name not in self.tools:
            result = {
                "success": False,
                "error": f"Tool does not exist: {name}",
            }

            return self._finalize_tool_call(
                tool_call_id=tool_call.id,
                tool_name=name,
                arguments=arguments,
                result=result,
                started_at=started_at,
            )

        if self.tool_history.repeated(name, arguments):
            result = {
                "success": False,
                "error": "Repeated tool call detected. Try another approach.",
            }

            return self._finalize_tool_call(
                tool_call_id=tool_call.id,
                tool_name=name,
                arguments=arguments,
                result=result,
                started_at=started_at,
            )

        tool = self.tools[name]

        try:  # 把arguments的TypeError和工具内部的TypeError bug分开。
            # 还有个点，python里参数的类型默认只是提示，不会做运行时校验，所以传入参数如果类型不对并不会触发TypeError。现在只会检测参数名是否正确、必须参数是否齐全、是否传入多余参数、位置参数和关键字参数是否冲突。
            signature = inspect.signature(tool.execute)
            signature.bind(**arguments)

        except TypeError as error:
            result = {"success": False, "error": f"Invalid tool parameters: {error}"}

            return self._finalize_tool_call(
                tool_call_id=tool_call.id,
                tool_name=name,
                arguments=arguments,
                result=result,
                started_at=started_at,
            )

        self.tool_history.add(name, arguments)  # 记录多少次参数合法的工具执行尝试

        try:
            result = tool.execute(**arguments)

        except Exception as error:
            result = {
                "success": False,
                "error": f"Tool execution failed: {error}",
            }

        if not isinstance(result, dict):
            result = {
                "success": False,
                "error": "Tool returned an invalid result: expected an object",
            }
        elif not isinstance(result.get("success"), bool):
            result = {
                "success": False,
                "error": "Tool returned an invalid result: missing boolean success",
            }

        return self._finalize_tool_call(
            tool_call_id=tool_call.id,
            tool_name=name,
            arguments=arguments,
            result=result,
            started_at=started_at,
        )
