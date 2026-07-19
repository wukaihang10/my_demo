import json
import time
import inspect
from typing import Any

from llm.client import chat, LLMClientError
from agent.trace import AgentTrace, StepTrace, ToolTrace
from agent.observation import process_observation
from agent.state import AgentState
from agent.task import TaskProfile
from agent.context import build_state_context
from agent.history import ToolHistory

from agent.plan import PlanError
from agent.planner import LLMPlanner, PlannerError
from agent.plan_evaluator import LLMPlanProgressEvaluator, PlanEvaluationError
from agent.plan_update import PlanController, PlanUpdateError, PlanUpdatePolicy
from agent.final_answer import FinalAnswerPolicy
from agent.stagnation import StagnationDecision, StagnationPolicy, StagnationTracker
from agent.outcome import AgentRunOutcome
from agent.execution import ToolBatchResult
from agent.config import AgentConfig, PlanningMode


class Agent:

    def __init__(
        self,
        task: TaskProfile[Any],
        planner: LLMPlanner | None = None,
        plan_evaluator: LLMPlanProgressEvaluator | None = None,
        plan_update_policy: PlanUpdatePolicy | None = None,
        final_answer_policy: FinalAnswerPolicy | None = None,
        stagnation_policy: StagnationPolicy | None = None,
        config: AgentConfig | None = None,
    ):
        self.task = task

        self.tools = {tool.name: tool for tool in self.task.tools}

        self.tool_schemas = [tool.schema() for tool in self.task.tools]

        self.trace = AgentTrace()

        self.state: AgentState[Any] = AgentState(
            task_state=self.task.create_state({}),
        )
        self.tool_history = ToolHistory()

        self.planner = planner or LLMPlanner()
        self.plan_evaluator = plan_evaluator or LLMPlanProgressEvaluator()
        self.plan_update_policy = plan_update_policy or PlanUpdatePolicy()
        self.plan_controller = PlanController(policy=self.plan_update_policy)

        self.final_answer_policy = final_answer_policy or FinalAnswerPolicy()

        self.stagnation_policy = stagnation_policy or StagnationPolicy()
        self.stagnation_tracker = StagnationTracker()

        self.last_outcome: AgentRunOutcome | None = None

        self.config = config or AgentConfig()

    def _preview(self, value, max_chars: int = 500) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)

        if len(text) <= max_chars:
            return text

        return text[:max_chars] + "...[truncated]"

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

        if not success and error_message is not None:
            self.state.add_error(error_message)

        self.task.reduce_tool_result(
            self.state.task_state,
            tool_name,
            arguments,
            normalized_result,
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
        if not self._uses_stagnation_recovery():
            return None

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

        return self._fail_run(
            error=error_message,
            stop_reason=decision.stop_reason or "stagnation_detected",
            step_trace=step_trace,
        )

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

    def _add_step_trace_once(
        self,
        step_trace: StepTrace | None,
    ) -> None:
        if step_trace is None:
            return

        if self.trace.steps and self.trace.steps[-1] is step_trace:
            return

        self.trace.add_step(step_trace)

    def _finalize_run(
        self,
        *,
        outcome: AgentRunOutcome,
        step_trace: StepTrace | None = None,
    ) -> str:
        self._add_step_trace_once(step_trace)

        self.last_outcome = outcome

        plan = self.state.plan

        if outcome.success:
            self.state.status = "completed"

            if plan is not None:
                if plan.status == "failed":
                    raise RuntimeError(
                        "A run cannot complete successfully "
                        "while its plan is failed."
                    )

                plan.finish(result=outcome.answer)

        else:
            assert outcome.error is not None

            self.state.status = "failed"
            self.state.add_error(outcome.error)

            if step_trace is not None:
                if step_trace.error is None:
                    step_trace.error = outcome.error

            if plan is not None and plan.status == "in_progress":
                plan.fail(outcome.error)

        self.trace.finish(outcome.stop_reason)

        return outcome.response

    def _complete_run(
        self,
        *,
        answer: str,
        step_trace: StepTrace | None = None,
        stop_reason: str = "completed",
    ) -> str:
        outcome = AgentRunOutcome.completed(
            answer=answer,
            stop_reason=stop_reason,
        )

        return self._finalize_run(
            outcome=outcome,
            step_trace=step_trace,
        )

    def _fail_run(
        self, *, error: str, stop_reason: str, step_trace: StepTrace | None = None
    ) -> str:
        outcome = AgentRunOutcome.failed(
            error=error,
            stop_reason=stop_reason,
        )

        return self._finalize_run(
            outcome=outcome,
            step_trace=step_trace,
        )

    def _reset_run_state(
        self,
        *,
        task_input: dict[str, Any],
        max_steps: int,
        max_tool_calls: int,
    ) -> None:
        self.trace = AgentTrace(
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
        )

        self.state = AgentState(
            task_state=self.task.create_state(task_input),
        )

        self.tool_history = ToolHistory()

        self.plan_controller = PlanController(
            policy=self.plan_update_policy,
        )

        self.stagnation_tracker = StagnationTracker()

        self.last_outcome = None

    def _initialize_plan(
        self,
        user_input: str,
    ) -> None:
        mode = self.config.planning_mode

        if mode is PlanningMode.NONE:
            self.state.plan = None
            return

        plan = self.planner.create_plan(user_input)

        if len(plan.steps) > self.plan_update_policy.max_total_steps:
            # 默认配置下这个条件不会触发，Planner最多6步，PlanUpdatePolicy最多10步
            raise PlannerError(
                f"The initial plan contains {len(plan.steps)} steps, but the runtime plan limit is {self.plan_update_policy.max_total_steps}"
            )

        plan.start()
        self.state.plan = plan

    def _uses_dynamic_planning(self) -> bool:
        return self.config.planning_mode is PlanningMode.DYNAMIC

    def _uses_final_answer_guard(self) -> bool:
        return self.config.enable_final_answer_guard

    def _uses_stagnation_recovery(self) -> bool:
        return self.config.enable_stagnation_recovery

    def run(
        self,
        user_input: str,
        max_steps: int = 10,
        max_tool_calls: int = 30,
        repo_url: str | None = None,
        task_input: dict[str, Any] | None = None,
    ) -> str:
        initial_task_input = dict(task_input or {})

        # Temporary compatibility for existing repository callers.
        if repo_url is not None:
            initial_task_input.setdefault(
                "repo_url",
                repo_url,
            )

        self._reset_run_state(
            task_input=initial_task_input,
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
        )

        if max_steps <= 0:
            error_message = "max_steps must be greater than zero."

            return self._fail_run(
                error=error_message,
                stop_reason="invalid_max_steps",
            )

        if max_tool_calls <= 0:
            error_message = "max_tool_calls must be greater than zero."

            return self._fail_run(
                error=error_message,
                stop_reason="invalid_max_tool_calls",
            )

        self.state.status = "running"

        try:
            self._initialize_plan(user_input)
        except PlannerError as error:
            return self._fail_run(
                error=f"Planning failed: {error}",
                stop_reason="planning_failed",
            )

        messages = [
            {"role": "system", "content": self.task.system_prompt},
            {"role": "user", "content": user_input},
        ]

        for step_number in range(1, max_steps + 1):
            print(f"\n===== Step {step_number} =====")

            step_trace = StepTrace(step=step_number)

            state_context = build_state_context(
                self.state,
                self.task,
                self.config.planning_mode,
            )

            request_messages = [*messages, {"role": "system", "content": state_context}]
            try:
                response = chat(
                    request_messages,
                    self.tool_schemas,
                )
            except LLMClientError as error:
                error_message = f"LLM request failed at step {step_number}: {error}"

                return self._fail_run(
                    error=error_message,
                    stop_reason="llm_error",
                    step_trace=step_trace,
                )

            if not response.tool_calls:
                candidate_response = response.content or ""

                if not self._uses_final_answer_guard():
                    step_trace.final_response = candidate_response

                    return self._complete_run(
                        answer=candidate_response,
                        step_trace=step_trace,
                    )

                decision = self.final_answer_policy.evaluate(
                    plan=self.state.plan,
                    tool_calls_used=self.trace.tool_calls_used,
                    response_content=candidate_response,
                )

                step_trace.final_answer_decision = decision.to_dict()

                if decision.allowed:
                    step_trace.final_response = candidate_response

                    return self._complete_run(
                        answer=candidate_response,
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
                        "content": (
                            self._build_final_answer_rejection_message(decision.reason)
                        ),
                    }
                )

                if self._uses_stagnation_recovery():
                    self.stagnation_tracker.record_final_answer_rejection(
                        self._get_current_plan_step_id()
                    )

                    stagnation_decision = self._evaluate_stagnation(step_trace)

                    assert stagnation_decision is not None

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
                continue

            # 下面是工具调用循环
            messages.append(response.model_dump(exclude_none=True))

            latest_evidence: list[dict[str, Any]] = []

            for tool_call in response.tool_calls:
                # 检查工具调用次数是否超了

                if self.trace.tool_calls_used >= max_tool_calls:
                    error_message = "The agent reached the maximum number of allowed tool calls before producing a final answer."

                    return self._fail_run(
                        error=error_message,
                        stop_reason="tool_budget_exceeded",
                        step_trace=step_trace,
                    )

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

            if self._uses_dynamic_planning() and self._uses_stagnation_recovery():
                self.stagnation_tracker.record_tool_batch(step_id_before_evaluation)

            if self._uses_dynamic_planning():
                step_trace.plan_update = self._evaluate_plan_progress(latest_evidence)

                plan_update_result = step_trace.plan_update

                if self._uses_stagnation_recovery():
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
                                    current_step_id=(self._get_current_plan_step_id()),
                                )

                plan = self.state.plan

                if plan is not None and plan.status == "failed":
                    plan_error = (
                        plan.error or "The task plan failed without an error message."
                    )

                    return self._fail_run(
                        error=f"The task plan failed: {plan_error}",
                        stop_reason="plan_failed",
                        step_trace=step_trace,
                    )

            if self._uses_stagnation_recovery():
                stagnation_decision = self._evaluate_stagnation(step_trace)

                assert stagnation_decision is not None

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
