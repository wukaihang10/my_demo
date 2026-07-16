import json
from types import SimpleNamespace
from typing import Any

import agent.agent as agent_module
from agent.agent import Agent
from llm.client import LLMClientError
from agent.plan import AgentPlan, PlanStepSpec
from agent.plan_evaluator import PlanEvaluationError
from agent.plan_update import (
    PlanUpdate,
    PlanUpdatePolicy,
)
from agent.final_answer import FinalAnswerPolicy
from agent.stagnation import StagnationPolicy


class FakePlanner:
    def create_plan(
        self,
        goal: str,
    ) -> AgentPlan:
        return AgentPlan.from_step_specs(
            goal=goal,
            step_specs=[
                PlanStepSpec(
                    description=("Gather enough evidence to answer " "the user."),
                    completion_criteria=(
                        "The required evidence has been " "collected."
                    ),
                )
            ],
        )


class FakePlanEvaluator:
    def __init__(
        self,
        updates: list[PlanUpdate],
    ) -> None:
        self._updates = iter(updates)
        self.calls: list[dict[str, Any]] = []

    def evaluate_progress(
        self,
        **kwargs: Any,
    ) -> PlanUpdate:
        self.calls.append(kwargs)

        try:
            return next(self._updates)
        except StopIteration as error:
            raise AssertionError(
                "The plan evaluator was called more times " "than expected."
            ) from error


class UnexpectedPlanEvaluator:
    def evaluate_progress(
        self,
        **kwargs: Any,
    ) -> PlanUpdate:
        raise AssertionError("The plan evaluator should not have been called.")


class FakeMessage:
    """A minimal replacement for an LLM assistance message."""

    def __init__(
        self, content: str | None = None, tool_calls: list[Any] | None = None
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {"role": "assistant", "content": self.content}

        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in self.tool_calls
            ]

        if exclude_none:
            data = {key: value for key, value in data.items() if value is not None}

        return data


class FakeTool:
    """A controllable tool used only by unit tests."""

    def __init__(
        self,
        result: dict[str, Any],
    ) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def execute(self, **arguments: Any) -> dict[str, Any]:
        self.calls.append(arguments)

        return self.result


def make_tool_call(
    name: str,
    arguments: dict[str, Any],
    call_id: str = "call_1",
):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments),
        ),
    )


def test_run_returns_direct_final_answer(
    monkeypatch,
) -> None:
    def fake_chat(messages, tools):
        return FakeMessage(content="Repository analysis completed.")

    monkeypatch.setattr(
        agent_module,
        "chat",
        fake_chat,
    )

    agent = Agent(
        planner=FakePlanner(),
        plan_evaluator=UnexpectedPlanEvaluator(),
    )

    answer = agent.run(
        user_input="Analyze the repository.",
        max_steps=3,
        max_tool_calls=3,
        repo_url="https://github.com/example/demo",
    )

    assert answer == "Repository analysis completed."

    assert agent.state.phase == "completed"
    assert agent.trace.status == "completed"

    assert agent.trace.steps_used == 1
    assert agent.trace.tool_calls_used == 0

    assert len(agent.trace.steps) == 1
    assert agent.trace.steps[0].final_response == "Repository analysis completed."

    assert agent.state.plan is not None

    assert agent.state.plan.status == "completed"
    assert agent.state.plan.result == ("Repository analysis completed.")
    assert all(step.status == "skipped" for step in agent.state.plan.steps)

    step = agent.trace.steps[0]

    assert step.final_answer_decision is not None
    assert step.final_answer_decision["allowed"] is True
    assert step.final_response == ("Repository analysis completed.")


def test_run_executes_tool_then_returns_answer(
    monkeypatch,
) -> None:
    tool_call = make_tool_call(
        name="fake_tool",
        arguments={"value": 42},
    )

    responses = iter(
        [
            FakeMessage(
                content=None,
                tool_calls=[tool_call],
            ),
            FakeMessage(content="The tool returned the expected result."),
        ]
    )

    received_messages: list[list[dict[str, Any]]] = []

    def fake_chat(messages, tools):
        received_messages.append(messages)

        return next(responses)

    monkeypatch.setattr(
        agent_module,
        "chat",
        fake_chat,
    )

    fake_tool = FakeTool(
        result={
            "success": True,
            "value": "tool result",
        }
    )

    evaluator = FakePlanEvaluator(
        updates=[
            PlanUpdate.from_dict(
                {
                    "action": "complete_current_step",
                    "result": ("The required tool evidence was collected."),
                }
            )
        ]
    )

    agent = Agent(planner=FakePlanner(), plan_evaluator=evaluator)
    agent.tools = {
        "fake_tool": fake_tool,
    }

    answer = agent.run(
        user_input="Use the fake tool.",
        max_steps=3,
        max_tool_calls=3,
    )

    assert fake_tool.calls == [{"value": 42}]

    assert agent.trace.status == "completed"
    assert agent.trace.steps_used == 2
    assert agent.trace.tool_calls_used == 1

    assert len(agent.trace.steps) == 2
    assert len(agent.trace.steps[0].tool_calls) == 1

    tool_trace = agent.trace.steps[0].tool_calls[0]

    assert tool_trace.tool_name == "fake_tool"
    assert tool_trace.success is True
    assert tool_trace.arguments == {"value": 42}

    second_request = received_messages[1]

    tool_messages = [
        message for message in second_request if message.get("role") == "tool"
    ]

    assert len(tool_messages) == 1

    observation = json.loads(tool_messages[0]["content"])

    assert observation["success"] is True
    assert observation["value"] == "tool result"

    assert len(evaluator.calls) == 1

    evidence = evaluator.calls[0]["latest_evidence"]

    assert len(evidence) == 1
    assert evidence[0]["tool_name"] == "fake_tool"
    assert evidence[0]["success"] is True
    assert evidence[0]["result"]["value"] == ("tool result")

    first_step = agent.trace.steps[0]

    assert first_step.plan_update is not None
    assert first_step.plan_update["success"] is True
    assert first_step.plan_update["action"] == ("complete_current_step")

    assert agent.state.plan is not None
    assert agent.state.plan.status == "completed"
    assert agent.state.plan.steps[0].status == "completed"
    assert agent.state.plan.result == ("The tool returned the expected result.")


def test_run_stops_when_tool_budget_is_exhausted(
    monkeypatch,
) -> None:
    first_call = make_tool_call(
        name="fake_tool",
        arguments={"value": 1},
        call_id="call_1",
    )

    second_call = make_tool_call(
        name="fake_tool",
        arguments={"value": 2},
        call_id="call_2",
    )

    def fake_chat(messages, tools):
        return FakeMessage(
            tool_calls=[
                first_call,
                second_call,
            ]
        )

    monkeypatch.setattr(
        agent_module,
        "chat",
        fake_chat,
    )

    fake_tool = FakeTool(
        result={
            "success": True,
            "value": "ok",
        }
    )

    agent = Agent(planner=FakePlanner(), plan_evaluator=UnexpectedPlanEvaluator())
    agent.tools = {
        "fake_tool": fake_tool,
    }

    answer = agent.run(
        user_input="Call tools.",
        max_steps=3,
        max_tool_calls=1,
    )

    assert "tool" in answer.lower()

    assert agent.trace.status == ("tool_budget_exceeded")

    assert agent.trace.tool_calls_used == 1

    trace_data = agent.trace.to_dict()
    assert trace_data["tool_calls_remaining"] == 0

    # Only the first tool is allowed to execute.
    assert fake_tool.calls == [{"value": 1}]

    assert len(agent.trace.steps) == 1
    assert len(agent.trace.steps[0].tool_calls) == 1

    assert agent.state.phase == "failed"
    assert agent.state.plan is not None
    assert agent.state.plan.status == "failed"
    assert agent.state.plan.current_step is not None
    assert agent.state.plan.current_step.status == "failed"

    assert len(agent.state.errors) == 1


def test_run_records_llm_error(
    monkeypatch,
) -> None:
    def fake_chat(messages, tools):
        raise LLMClientError("simulated connection failure")

    monkeypatch.setattr(
        agent_module,
        "chat",
        fake_chat,
    )

    agent = Agent(planner=FakePlanner(), plan_evaluator=UnexpectedPlanEvaluator())

    answer = agent.run(
        user_input="Analyze repository.",
        max_steps=3,
        max_tool_calls=3,
    )

    assert "failed" in answer.lower()

    assert agent.state.phase == "failed"
    assert agent.trace.status == "llm_error"

    assert agent.trace.steps_used == 1
    assert agent.trace.tool_calls_used == 0

    assert len(agent.state.errors) == 1
    assert "simulated connection failure" in (agent.state.errors[0])

    step = agent.trace.steps[0]

    assert step.error is not None
    assert "simulated connection failure" in (step.error)

    assert agent.state.plan is not None
    assert agent.state.plan.status == "failed"
    assert agent.state.plan.current_step is not None
    assert agent.state.plan.current_step.status == "failed"

    assert "simulated connection failure" in (agent.state.plan.current_step.error or "")


def test_run_rejects_invalid_max_steps() -> None:
    agent = Agent()

    answer = agent.run(
        user_input="Analyze repository.",
        max_steps=0,
        max_tool_calls=3,
    )

    assert answer == "max_steps must be greater than zero."

    assert agent.trace.status == "invalid_max_steps"
    assert agent.state.phase == "failed"

    assert agent.state.plan is None


def test_run_rejects_invalid_tool_budget() -> None:
    agent = Agent()

    answer = agent.run(
        user_input="Analyze repository.",
        max_steps=3,
        max_tool_calls=0,
    )

    assert answer == "max_tool_calls must be greater than zero."

    assert agent.trace.status == "invalid_max_tool_calls"
    assert agent.state.phase == "failed"

    assert agent.state.plan is None


def test_run_evaluates_plan_once_per_tool_batch(
    monkeypatch,
) -> None:
    first_call = make_tool_call(
        name="fake_tool",
        arguments={"value": 1},
        call_id="call_1",
    )
    second_call = make_tool_call(
        name="fake_tool",
        arguments={"value": 2},
        call_id="call_2",
    )

    responses = iter(
        [
            FakeMessage(
                tool_calls=[
                    first_call,
                    second_call,
                ]
            ),
            FakeMessage(content="Both results were collected."),
        ]
    )

    def fake_chat(messages, tools):
        return next(responses)

    monkeypatch.setattr(
        agent_module,
        "chat",
        fake_chat,
    )

    evaluator = FakePlanEvaluator(
        updates=[
            PlanUpdate.from_dict(
                {
                    "action": "complete_current_step",
                    "result": ("Both tool calls completed."),
                }
            )
        ]
    )

    fake_tool = FakeTool(
        result={
            "success": True,
            "value": "ok",
        }
    )

    agent = Agent(
        planner=FakePlanner(),
        plan_evaluator=evaluator,
    )
    agent.tools = {
        "fake_tool": fake_tool,
    }

    answer = agent.run(
        user_input="Call the tool twice.",
        max_steps=3,
        max_tool_calls=3,
    )

    assert answer == "Both results were collected."
    assert fake_tool.calls == [
        {"value": 1},
        {"value": 2},
    ]

    assert len(evaluator.calls) == 1

    evidence = evaluator.calls[0]["latest_evidence"]

    assert len(evidence) == 2
    assert evidence[0]["tool_call_id"] == "call_1"
    assert evidence[1]["tool_call_id"] == "call_2"

    assert agent.trace.steps[0].plan_update is not None
    assert agent.trace.steps[0].plan_update["action"] == "complete_current_step"


def test_plan_evaluation_error_is_nonfatal(
    monkeypatch,
) -> None:
    tool_call = make_tool_call(
        name="fake_tool",
        arguments={"value": 42},
    )

    responses = iter(
        [
            FakeMessage(tool_calls=[tool_call]),
            FakeMessage(content="The task still completed."),
            FakeMessage(content="The task still completed."),
        ]
    )

    def fake_chat(messages, tools):
        return next(responses)

    monkeypatch.setattr(
        agent_module,
        "chat",
        fake_chat,
    )

    class ErrorEvaluator:
        def evaluate_progress(
            self,
            **kwargs: Any,
        ) -> PlanUpdate:
            raise PlanEvaluationError("simulated invalid evaluator response")

    fake_tool = FakeTool(
        result={
            "success": True,
            "value": "ok",
        }
    )

    agent = Agent(
        planner=FakePlanner(),
        plan_evaluator=ErrorEvaluator(),
        stagnation_policy=StagnationPolicy(
            max_recovery_attempts_per_step=0,
        ),
    )
    agent.tools = {
        "fake_tool": fake_tool,
    }

    answer = agent.run(
        user_input="Use the tool.",
        max_steps=3,
        max_tool_calls=3,
    )

    assert (
        answer
        == "Agent execution stagnated: The executor repeatedly proposed a final answer while the task plan was incomplete. The recovery allowance for the current plan step has been exhausted."
    )
    assert agent.state.phase == "failed"
    assert agent.trace.status == "stagnation_rejected_answers"

    assert any(
        "Plan progress evaluation failed" in error for error in agent.state.errors
    )

    update_trace = agent.trace.steps[0].plan_update

    assert update_trace is not None
    assert update_trace["success"] is False
    assert "simulated invalid evaluator response" in update_trace["error"]


def test_failed_plan_stops_agent_after_tool_batch(
    monkeypatch,
) -> None:
    tool_call = make_tool_call(
        name="fake_tool",
        arguments={"value": 42},
    )

    executor_call_count = 0

    def fake_chat(messages, tools):
        nonlocal executor_call_count
        executor_call_count += 1

        return FakeMessage(tool_calls=[tool_call])

    monkeypatch.setattr(
        agent_module,
        "chat",
        fake_chat,
    )

    evaluator = FakePlanEvaluator(
        updates=[
            PlanUpdate.from_dict(
                {
                    "action": "fail_current_step",
                    "error": ("The required information is " "unavailable."),
                }
            )
        ]
    )

    fake_tool = FakeTool(
        result={
            "success": False,
            "error": "Repository is unavailable.",
        }
    )

    agent = Agent(
        planner=FakePlanner(),
        plan_evaluator=evaluator,
    )
    agent.tools = {
        "fake_tool": fake_tool,
    }

    answer = agent.run(
        user_input="Inspect the unavailable repository.",
        max_steps=3,
        max_tool_calls=3,
    )

    assert "task plan failed" in answer.lower()
    assert "required information is unavailable" in answer.lower()

    assert executor_call_count == 1
    assert agent.state.phase == "failed"
    assert agent.trace.status == "plan_failed"

    assert agent.state.plan is not None
    assert agent.state.plan.status == "failed"
    assert agent.state.plan.current_step.status == "failed"


def test_run_rejects_premature_final_answer_after_tool_use(
    monkeypatch,
) -> None:
    first_tool_call = make_tool_call(
        name="fake_tool",
        arguments={"value": 1},
        call_id="call_1",
    )

    second_tool_call = make_tool_call(
        name="fake_tool",
        arguments={"value": 2},
        call_id="call_2",
    )

    responses = iter(
        [
            FakeMessage(
                tool_calls=[first_tool_call],
            ),
            FakeMessage(
                content=(
                    "This answer is premature because " "the plan is not complete."
                ),
            ),
            FakeMessage(
                tool_calls=[second_tool_call],
            ),
            FakeMessage(
                content="The final verified answer.",
            ),
        ]
    )

    received_messages: list[list[dict[str, Any]]] = []

    def fake_chat(messages, tools):
        received_messages.append(messages)
        return next(responses)

    monkeypatch.setattr(
        agent_module,
        "chat",
        fake_chat,
    )

    evaluator = FakePlanEvaluator(
        updates=[
            PlanUpdate.from_dict(
                {
                    "action": "keep_current_step",
                    "reason": ("More evidence is required."),
                }
            ),
            PlanUpdate.from_dict(
                {
                    "action": ("complete_current_step"),
                    "result": ("All required evidence was " "collected."),
                }
            ),
        ]
    )

    fake_tool = FakeTool(
        result={
            "success": True,
            "value": "tool result",
        }
    )

    agent = Agent(
        planner=FakePlanner(),
        plan_evaluator=evaluator,
    )
    agent.tools = {
        "fake_tool": fake_tool,
    }

    answer = agent.run(
        user_input="Complete the tool-assisted task.",
        max_steps=4,
        max_tool_calls=3,
    )

    assert answer == "The final verified answer."

    assert fake_tool.calls == [
        {"value": 1},
        {"value": 2},
    ]

    assert len(evaluator.calls) == 2
    assert len(agent.trace.steps) == 4

    rejected_step = agent.trace.steps[1]

    assert rejected_step.final_answer_decision is not None
    assert rejected_step.final_answer_decision["allowed"] is False
    assert rejected_step.final_response is None

    accepted_step = agent.trace.steps[3]

    assert accepted_step.final_answer_decision is not None
    assert accepted_step.final_answer_decision["allowed"] is True
    assert accepted_step.final_response == ("The final verified answer.")

    # The third executor request should contain
    # feedback explaining that the previous answer
    # was rejected.
    third_request = received_messages[2]

    rejection_messages = [
        message
        for message in third_request
        if (
            message.get("role") == "system"
            and "not accepted as the final answer" in message.get("content", "")
        )
    ]

    assert len(rejection_messages) == 1

    assert agent.state.plan is not None
    assert agent.state.plan.status == "completed"
    assert agent.state.plan.result == ("The final verified answer.")


def test_agent_can_disable_direct_final_answers(
    monkeypatch,
) -> None:
    responses = iter(
        [
            FakeMessage(content="An unsupported direct answer."),
            FakeMessage(content="Another unsupported answer."),
        ]
    )

    def fake_chat(messages, tools):
        return next(responses)

    monkeypatch.setattr(
        agent_module,
        "chat",
        fake_chat,
    )

    agent = Agent(
        planner=FakePlanner(),
        plan_evaluator=UnexpectedPlanEvaluator(),
        final_answer_policy=FinalAnswerPolicy(
            allow_direct_answer_before_tool_use=False,
        ),
        stagnation_policy=StagnationPolicy(
            max_recovery_attempts_per_step=0,
        )
    )

    answer = agent.run(
        user_input="Analyze an external repository.",
        max_steps=2,
        max_tool_calls=3,
    )

    assert "he executor repeatedly proposed a final answer" in answer.lower()
    assert agent.trace.status == "stagnation_rejected_answers"

    assert len(agent.trace.steps) == 2

    assert all(
        step.final_answer_decision is not None
        and (step.final_answer_decision["allowed"] is False)
        for step in agent.trace.steps
    )


def test_agent_recovers_once_then_stops_after_more_keeps(
    monkeypatch,
) -> None:
    first_tool_call = make_tool_call(
        name="fake_tool",
        arguments={"value": 1},
        call_id="call_1",
    )

    second_tool_call = make_tool_call(
        name="fake_tool",
        arguments={"value": 2},
        call_id="call_2",
    )

    responses = iter(
        [
            FakeMessage(
                tool_calls=[first_tool_call],
            ),
            FakeMessage(
                tool_calls=[second_tool_call],
            ),
        ]
    )

    received_messages: list[list[dict[str, Any]]] = []

    def fake_chat(messages, tools):
        received_messages.append(messages)
        return next(responses)

    monkeypatch.setattr(
        agent_module,
        "chat",
        fake_chat,
    )

    evaluator = FakePlanEvaluator(
        updates=[
            PlanUpdate.from_dict(
                {
                    "action": "keep_current_step",
                    "reason": "More evidence is needed.",
                }
            ),
            PlanUpdate.from_dict(
                {
                    "action": "keep_current_step",
                    "reason": "Still not enough evidence.",
                }
            ),
        ]
    )

    fake_tool = FakeTool(
        result={
            "success": True,
            "value": "ok",
        }
    )

    agent = Agent(
        planner=FakePlanner(),
        plan_evaluator=evaluator,
        stagnation_policy=StagnationPolicy(
            max_consecutive_keeps=1,
            max_attempts_per_step=10,
            max_recovery_attempts_per_step=1,
        ),
    )

    agent.tools = {
        "fake_tool": fake_tool,
    }

    result = agent.run(
        user_input="Complete the task.",
        max_steps=10,
        max_tool_calls=10,
    )

    assert "execution stagnated" in result.lower()

    assert fake_tool.calls == [
        {"value": 1},
        {"value": 2},
    ]

    assert agent.trace.status == ("stagnation_consecutive_keeps")

    first_trace = agent.trace.steps[0]

    assert first_trace.stagnation is not None
    assert first_trace.stagnation["decision"]["should_recover"] is True
    assert first_trace.stagnation["recovery"]["applied"] is True
    assert first_trace.stagnation["recovery"]["attempt"] == 1

    second_request = received_messages[1]

    recovery_messages = [
        message
        for message in second_request
        if (
            message.get("role") == "system"
            and "Change the execution strategy materially" in message.get("content", "")
        )
    ]

    assert len(recovery_messages) == 1

    second_trace = agent.trace.steps[1]

    assert second_trace.stagnation is not None
    assert second_trace.stagnation["decision"]["should_stop"] is True


def test_agent_stops_after_repeated_premature_answers(
    monkeypatch,
) -> None:
    tool_call = make_tool_call(
        name="fake_tool",
        arguments={"value": 1},
        call_id="call_1",
    )

    responses = iter(
        [
            FakeMessage(tool_calls=[tool_call]),
            FakeMessage(content="Premature answer one."),
            FakeMessage(content="Premature answer two."),
        ]
    )

    def fake_chat(messages, tools):
        return next(responses)

    monkeypatch.setattr(
        agent_module,
        "chat",
        fake_chat,
    )

    evaluator = FakePlanEvaluator(
        updates=[
            PlanUpdate.from_dict(
                {
                    "action": "keep_current_step",
                    "reason": "More evidence is required.",
                }
            )
        ]
    )

    agent = Agent(
        planner=FakePlanner(),
        plan_evaluator=evaluator,
        stagnation_policy=StagnationPolicy(
            max_consecutive_rejected_final_answers=2,
            max_consecutive_keeps=10,
            max_attempts_per_step=10,
            max_recovery_attempts_per_step=0,
        ),
    )

    agent.tools = {
        "fake_tool": FakeTool(
            result={
                "success": True,
                "value": "partial evidence",
            }
        )
    }

    result = agent.run(
        user_input="Complete the task.",
        max_steps=10,
        max_tool_calls=10,
    )

    assert "execution stagnated" in result.lower()
    assert agent.trace.status == ("stagnation_rejected_answers")

    assert len(agent.trace.steps) == 3

    final_trace = agent.trace.steps[-1]

    assert final_trace.final_answer_decision is not None
    assert final_trace.final_answer_decision["allowed"] is False

    assert final_trace.stagnation is not None
    assert final_trace.stagnation["snapshot"]["consecutive_rejected_final_answers"] == 2


def test_agent_recovers_from_stagnation_and_completes(
    monkeypatch,
) -> None:
    first_tool_call = make_tool_call(
        name="fake_tool",
        arguments={"value": 1},
        call_id="call_1",
    )

    second_tool_call = make_tool_call(
        name="fake_tool",
        arguments={"value": 2},
        call_id="call_2",
    )

    responses = iter(
        [
            FakeMessage(
                tool_calls=[first_tool_call],
            ),
            FakeMessage(
                tool_calls=[second_tool_call],
            ),
            FakeMessage(content="The recovered task completed."),
        ]
    )

    received_messages: list[list[dict[str, Any]]] = []

    def fake_chat(messages, tools):
        received_messages.append(messages)
        return next(responses)

    monkeypatch.setattr(
        agent_module,
        "chat",
        fake_chat,
    )

    evaluator = FakePlanEvaluator(
        updates=[
            PlanUpdate.from_dict(
                {
                    "action": "keep_current_step",
                    "reason": (
                        "The first strategy did not provide " "enough evidence."
                    ),
                }
            ),
            PlanUpdate.from_dict(
                {
                    "action": "complete_current_step",
                    "result": (
                        "The alternative strategy provided " "the required evidence."
                    ),
                }
            ),
        ]
    )

    fake_tool = FakeTool(
        result={
            "success": True,
            "value": "evidence",
        }
    )

    agent = Agent(
        planner=FakePlanner(),
        plan_evaluator=evaluator,
        stagnation_policy=StagnationPolicy(
            max_consecutive_keeps=1,
            max_attempts_per_step=10,
            max_recovery_attempts_per_step=1,
        ),
    )

    agent.tools = {
        "fake_tool": fake_tool,
    }

    result = agent.run(
        user_input="Complete the task.",
        max_steps=5,
        max_tool_calls=5,
    )

    assert result == "The recovered task completed."

    assert fake_tool.calls == [
        {"value": 1},
        {"value": 2},
    ]

    assert agent.state.phase == "completed"
    assert agent.trace.status == "completed"

    assert agent.state.plan is not None
    assert agent.state.plan.status == "completed"
    assert agent.state.plan.result == ("The recovered task completed.")

    first_trace = agent.trace.steps[0]

    assert first_trace.stagnation is not None
    assert first_trace.stagnation["decision"]["should_recover"] is True

    second_trace = agent.trace.steps[1]

    assert second_trace.plan_update is not None
    assert second_trace.plan_update["action"] == "complete_current_step"

    second_request = received_messages[1]

    assert any(
        message.get("role") == "system"
        and "different tool, different arguments" in message.get("content", "")
        for message in second_request
    )
