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

from agent.planner import LLMPlanner, PlannerError
from agent.plan_evaluator import LLMPlanProgressEvaluator, PlanEvaluationError
from agent.plan_update import PlanController, PlanUpdateError, PlanUpdatePolicy
from agent.final_answer import FinalAnswerPolicy
from agent.stagnation import StagnationDecision, StagnationPolicy, StagnationTracker

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
    def __init__(
        self,
        planner: LLMPlanner | None = None,
        plan_evaluator: LLMPlanProgressEvaluator | None = None,
        plan_update_policy: PlanUpdatePolicy | None = None,
        final_answer_policy: FinalAnswerPolicy | None = None,
        stagnation_policy: StagnationPolicy | None = None,
    ):
        self.tools = TOOL_MAP
        self.trace = AgentTrace()
        self.state = RepositoryState()
        self.tool_history = ToolHistory()

        self.planner = planner or LLMPlanner()
        self.plan_evaluator = plan_evaluator or LLMPlanProgressEvaluator()
        self.plan_update_policy = plan_update_policy or PlanUpdatePolicy()
        self.plan_controller = PlanController(policy=self.plan_update_policy)

        self.final_answer_policy = final_answer_policy or FinalAnswerPolicy()

        self.stagnation_policy = stagnation_policy or StagnationPolicy()
        self.stagnation_tracker = StagnationTracker()

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

    def _evaluate_plan_progress(
        self,
        latest_evidence: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        plan = self.state.plan

        if plan is None:
            return None

        if plan.status != "in_progress":
            return None

        if not latest_evidence:
            return None

        try:
            update = self.plan_evaluator.evaluate_progress(
                plan=plan,
                state=self.state,
                latest_evidence=latest_evidence,
                updates_remaining=self.plan_controller.updates_remaining,
                max_total_steps=self.plan_controller.policy.max_total_steps,
                max_added_steps_per_update=self.plan_controller.policy.max_added_steps_per_update,
            )

            return self.plan_controller.apply_update(
                plan,
                update,
            )

        except (PlanEvaluationError, PlanUpdateError) as error:
            error_message = f"Plan progress evaluation failed: {error}"

            self.state.add_error(error_message)

            return {"success": False, "error": error_message}

    def _build_final_answer_rejection_message(
        self,
        reason: str,
    ) -> str:
        plan = self.state.plan
        current_step = plan.current_step if plan is not None else None

        lines = [
            f"The previous response was not accepted as the final answer.\nReason: {reason}"
        ]

        if current_step:
            lines.extend(
                [
                    f"Continue working on the current plan step {current_step.id}: {current_step.description}",
                    "Do not repeat the same final answer until the plan has made progress.",
                ]
            )
        else:
            lines.append(
                "Continue the task and gather the missing information before answering."
            )

        return "\n".join(lines)

    def _get_current_plan_step_id(self) -> int | None:
        plan = self.state.plan

        if plan is None or plan.current_step is None:
            return None

        return plan.current_step.id

    def _evaluate_stagnation(
        self,
        step_trace: StepTrace,
    ) -> StagnationDecision:
        snapshot = self.stagnation_tracker.snapshot()

        decision = self.stagnation_policy.evaluate(snapshot)

        step_trace.stagnation = {
            "snapshot": snapshot.to_dict(),
            "decision": decision.to_dict(),
        }

        return decision

    def _stop_for_stagnation(
        self,
        *,
        decision: StagnationDecision,
        step_trace: StepTrace,
    ) -> str:
        message = (
            decision.message
            or "Agent executor stopped because no progress was being made."
        )

        error_message = f"Agent execution stagnated: {message}"

        self.state.add_error(error_message)
        self.state.phase = "failed"

        plan = self.state.plan

        if plan is not None:
            plan.fail(error_message)

        self.trace.add_step(step_trace)
        self.trace.finish(decision.stop_reason or "stagnation_detected")

        return error_message

    def _build_stagnation_recovery_message(
        self,
        *,
        decision: StagnationDecision,
        recovery_attempt: str,
    ) -> str:
        plan = self.state.plan
        current_step = plan.current_step if plan is not None else None

        lines = [
            "The current execution strategy is not making sufficient progress.",
            f"Detected problem: {decision.message}",
            f"Recovery attempt: {recovery_attempt}/{self.stagnation_policy.max_recovery_attempts_per_step}",
            "",
            "Change the execution strategy materially.",
            "Do not repeat the same tool with the same arguments unless new evidence justifies it.",
            "Inspect the existing observations and choose a different tool, different arguments, or a different source of evidence.",
            "Do not provide a final answer while the plan step remains incomplete.",
        ]

        if current_step is not None:
            lines.extend(
                [
                    "",
                    f"Current plan step {current_step.id}: {current_step.description}",
                ]
            )

            if current_step.completion_criteria:
                lines.append(f"Completion criteria: {current_step.completion_criteria}")

        return "\n".join(lines)

    def _apply_stagnation_recovery(
        self,
        *,
        decision: StagnationDecision,
        step_trace: StepTrace,
        messages: list[dict[str, Any]],
    ) -> None:
        step_id = self._get_current_plan_step_id()

        self.stagnation_tracker.record_recovery(step_id)

        snapshot_after_recovery = self.stagnation_tracker.snapshot()

        recovery_attempt = snapshot_after_recovery.recovery_attempts_on_current_step

        recovery_message = self._build_stagnation_recovery_message(
            decision=decision,
            recovery_attempt=recovery_attempt,
        )

        messages.append(
            {
                "role": "system",
                "content": recovery_message,
            }
        )

        if step_trace.stagnation is None:
            step_trace.stagnation = {}

        step_trace.stagnation["recovery"] = {
            "applied": True,
            "attempt": recovery_attempt,
            "trigger": decision.trigger,
            "snapshot_after_recovery": snapshot_after_recovery.to_dict(),
        }

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

        self.plan_controller = PlanController(
            policy=self.plan_update_policy,
        )

        self.stagnation_tracker = StagnationTracker()

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

            if len(plan.steps) > self.plan_update_policy.max_total_steps:
                # 默认配置下这个条件不会触发，Planner最多6步，PlanUpdatePolicy最多10步
                raise PlannerError(
                    f"The initial plan contains {len(plan.steps)} steps, but the runtime plan limit is {self.plan_update_policy.max_total_steps}"
                )

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
                candidate_response = response.content or ""

                decision = self.final_answer_policy.evaluate(
                    plan=self.state.plan,
                    tool_calls_used=self.trace.tool_calls_used,
                    response_content=candidate_response,
                )

                step_trace.final_answer_decision = decision.to_dict()

                if decision.allowed:
                    step_trace.final_response = candidate_response

                    self.trace.add_step(step_trace)
                    self.trace.finish("completed")

                    self.state.phase = "completed"

                    if self.state.plan is not None:
                        self.state.plan.finish(result=candidate_response)

                    return candidate_response

                self.stagnation_tracker.record_final_answer_rejection(
                    self._get_current_plan_step_id()
                )

                stagnation_decision = self._evaluate_stagnation(step_trace)

                if stagnation_decision.should_stop:
                    return self._stop_for_stagnation(
                        decision=stagnation_decision,
                        step_trace=step_trace,
                    )

                if candidate_response.strip():

                    messages.append(
                        {
                            "role": "assistant",
                            "content": candidate_response,
                        }
                    )

                messages.append(
                    {
                        "role": "system",
                        "content": self._build_final_answer_rejection_message(
                            decision.reason
                        ),
                    }
                )

                if stagnation_decision.should_recover:
                    self._apply_stagnation_recovery(
                        decision=stagnation_decision,
                        step_trace=step_trace,
                        messages=messages,
                    )

                self.trace.add_step(step_trace)
                continue

            # 下面是工具调用循环
            messages.append(response.model_dump(exclude_none=True))

            latest_evidence: list[dict[str, Any]] = []

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

                latest_evidence.append(
                    {
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_trace.tool_name,
                        "arguments": tool_trace.arguments,
                        "success": tool_trace.success,
                        "result": result,
                    }
                )

                content = process_observation(result)

                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": content}
                )

            step_id_before_evaluation = self._get_current_plan_step_id()

            self.stagnation_tracker.record_tool_batch(step_id_before_evaluation)

            step_trace.plan_update = self._evaluate_plan_progress(latest_evidence)

            plan_update_result = step_trace.plan_update

            if plan_update_result is not None:
                if plan_update_result.get("success") is False:
                    self.stagnation_tracker.record_evaluation_error(
                        self._get_current_plan_step_id()
                    )

                else:
                    action = plan_update_result.get("action")

                    if isinstance(action, str):
                        self.stagnation_tracker.record_plan_update(
                            action=action,
                            current_step_id=self._get_current_plan_step_id(),
                        )

            plan = self.state.plan

            if plan is not None and plan.status == "failed":
                plan_error = (
                    plan.error or "The task plan failed without an error message."
                )

                error_message = f"The task plan failed: {plan_error}"

                self.state.add_error(error_message)
                self.state.phase = "failed"

                self.trace.add_step(step_trace)
                self.trace.finish("plan_failed")

                return error_message

            stagnation_decision = self._evaluate_stagnation(step_trace)

            if stagnation_decision.should_stop:
                return self._stop_for_stagnation(
                    decision=stagnation_decision,
                    step_trace=step_trace,
                )

            if stagnation_decision.should_recover:
                self._apply_stagnation_recovery(
                    decision=stagnation_decision,
                    step_trace=step_trace,
                    messages=messages,
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
