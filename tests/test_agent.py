import json
from dataclasses import dataclass
from typing import Any, Callable

import pytest

import agent.agent as agent_module
from agent.agent import Agent
from agent.config import AgentConfig, PlanningMode
from agent.final_answer import FinalAnswerPolicy
from agent.outcome import AgentRunOutcome
from agent.plan import AgentPlan, PlanStepSpec
from agent.plan_update import PlanUpdate
from agent.state import RunStatus
from agent.task import TaskProfile
from llm.client import LLMClientError
from tools.base import Tool


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


@dataclass
class FakeMessage:
    content: str | None = None
    tool_calls: list[FakeToolCall] | None = None

    def __post_init__(self) -> None:
        if self.tool_calls is None:
            self.tool_calls = []


class FakePlanner:
    def __init__(self) -> None:
        self.goals: list[str] = []

    def create_plan(self, goal: str) -> AgentPlan:
        self.goals.append(goal)
        return AgentPlan.from_step_specs(
            goal=goal,
            step_specs=[
                PlanStepSpec(
                    description="Collect the required evidence.",
                    completion_criteria="Enough evidence has been collected.",
                )
            ],
        )


class FakePlanEvaluator:
    def __init__(self, *updates: PlanUpdate) -> None:
        self._updates = iter(updates)
        self.calls: list[dict[str, Any]] = []

    def evaluate_progress(self, **kwargs: Any) -> PlanUpdate:
        self.calls.append(kwargs)
        try:
            return next(self._updates)
        except StopIteration as error:
            raise AssertionError("Unexpected plan evaluation.") from error


def make_tool_call(
    name: str = "store_value",
    arguments: Any = None,
    call_id: str = "call_1",
) -> FakeToolCall:
    if arguments is None:
        arguments = {"value": 42}

    encoded_arguments = (
        arguments if isinstance(arguments, str) else json.dumps(arguments)
    )
    return FakeToolCall(
        id=call_id,
        function=FakeFunction(name=name, arguments=encoded_arguments),
    )


def make_task(
    function: Callable[..., dict[str, Any]] | None = None,
) -> TaskProfile[dict[str, Any]]:
    def store_value(value: int) -> dict[str, Any]:
        return {"success": True, "value": value}

    tool = Tool(
        name="store_value",
        description="Store one value.",
        function=function or store_value,
        parameters={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
    )

    def create_state(input_data: dict[str, Any]) -> dict[str, Any]:
        return {"label": input_data.get("label"), "values": []}

    def reduce_tool_result(
        state: dict[str, Any],
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        if tool_name == "store_value" and result.get("success") is True:
            state["values"].append(result["value"])

    def build_context(state: dict[str, Any]) -> str:
        return f"Label: {state['label']}\nStored values: {state['values']}"

    return TaskProfile(
        name="value_store",
        system_prompt="You store values for tests.",
        tools=(tool,),
        create_state=create_state,
        reduce_tool_result=reduce_tool_result,
        build_context=build_context,
    )


def install_responses(
    monkeypatch: pytest.MonkeyPatch,
    *responses: FakeMessage,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    response_iterator = iter(responses)
    requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    def fake_chat(messages, tools):
        requests.append((messages, tools))
        try:
            return next(response_iterator)
        except StopIteration as error:
            raise AssertionError("Unexpected LLM request.") from error

    monkeypatch.setattr(agent_module, "chat", fake_chat)
    return requests


def test_runtime_properties_require_an_active_run() -> None:
    agent = Agent(task=make_task(), config=AgentConfig(planning_mode=PlanningMode.NONE))

    assert agent.last_outcome is None
    with pytest.raises(RuntimeError, match="No active agent run"):
        _ = agent.state
    with pytest.raises(RuntimeError, match="No active agent run"):
        _ = agent.trace


def test_constructor_initializes_only_enabled_capabilities() -> None:
    direct_agent = Agent(
        task=make_task(),
        config=AgentConfig(planning_mode=PlanningMode.NONE),
    )

    assert direct_agent.planner is None
    assert direct_agent.plan_evaluator is None
    assert direct_agent.plan_controller is None
    assert direct_agent.final_answer_policy is None
    assert direct_agent.stagnation_policy is None

    dynamic_agent = Agent(
        task=make_task(),
        config=AgentConfig(
            planning_mode=PlanningMode.DYNAMIC,
            enable_final_answer_guard=True,
            enable_stagnation_recovery=True,
        ),
    )

    assert dynamic_agent.planner is not None
    assert dynamic_agent.plan_evaluator is not None
    assert dynamic_agent.plan_controller is not None
    assert dynamic_agent.final_answer_policy is not None
    assert dynamic_agent.stagnation_policy is not None


def test_stagnation_recovery_requires_a_signal_source() -> None:
    with pytest.raises(ValueError, match="requires dynamic planning"):
        Agent(
            task=make_task(),
            config=AgentConfig(
                planning_mode=PlanningMode.NONE,
                enable_stagnation_recovery=True,
            ),
        )


def test_run_without_planning_returns_direct_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = install_responses(monkeypatch, FakeMessage(content="Task completed."))
    agent = Agent(task=make_task(), config=AgentConfig(planning_mode=PlanningMode.NONE))

    answer = agent.run(
        user_input="Complete the task.",
        task_input={"label": "example"},
    )

    assert answer == "Task completed."
    assert agent.state.status is RunStatus.COMPLETED
    assert agent.state.plan is None
    assert agent.state.task_state == {"label": "example", "values": []}
    assert agent.last_outcome == AgentRunOutcome.completed(answer="Task completed.")
    assert agent.trace.steps_used == 1
    assert agent.trace.finished_at is not None

    messages, tool_schemas = requests[0]
    assert messages[0] == {"role": "system", "content": "You store values for tests."}
    assert messages[1] == {"role": "user", "content": "Complete the task."}
    assert "Task plan:" not in messages[-1]["content"]
    assert tool_schemas[0]["function"]["name"] == "store_value"


def test_static_planning_builds_a_plan_without_progress_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_responses(monkeypatch, FakeMessage(content="Static task completed."))
    planner = FakePlanner()
    evaluator = FakePlanEvaluator()
    agent = Agent(
        task=make_task(),
        planner=planner,
        plan_evaluator=evaluator,
        config=AgentConfig(planning_mode=PlanningMode.STATIC),
    )

    answer = agent.run("Follow the roadmap.")

    assert answer == "Static task completed."
    assert planner.goals == ["Follow the roadmap."]
    assert evaluator.calls == []
    assert agent.state.plan is not None
    assert agent.state.plan.status == "completed"
    assert agent.state.plan.result == "Static task completed."
    assert agent.state.plan.steps[0].status == "skipped"


def test_tool_result_updates_task_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = install_responses(
        monkeypatch,
        FakeMessage(tool_calls=[make_tool_call(arguments={"value": 7})]),
        FakeMessage(content="Stored."),
    )
    agent = Agent(task=make_task(), config=AgentConfig(planning_mode=PlanningMode.NONE))

    answer = agent.run("Store seven.", max_steps=2)

    assert answer == "Stored."
    assert agent.state.task_state["values"] == [7]
    assert agent.trace.tool_calls_used == 1
    assert agent.trace.steps[0].tool_calls[0].success is True
    assert agent.trace.steps[0].tool_calls[0].arguments == {"value": 7}

    second_request_messages, _ = requests[1]
    assert any(
        message.get("role") == "tool" and message.get("tool_call_id") == "call_1"
        for message in second_request_messages
    )
    assert "Stored values: [7]" in second_request_messages[-1]["content"]


def test_dynamic_planning_evaluates_each_tool_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_responses(
        monkeypatch,
        FakeMessage(tool_calls=[make_tool_call(arguments={"value": 9})]),
        FakeMessage(content="Verified result."),
    )
    evaluator = FakePlanEvaluator(
        PlanUpdate.from_dict(
            {
                "action": "complete_current_step",
                "result": "The value was collected.",
            }
        )
    )
    agent = Agent(
        task=make_task(),
        planner=FakePlanner(),
        plan_evaluator=evaluator,
        config=AgentConfig(planning_mode=PlanningMode.DYNAMIC),
    )

    answer = agent.run("Collect a value.", max_steps=2)

    assert answer == "Verified result."
    assert len(evaluator.calls) == 1
    evidence = evaluator.calls[0]["latest_evidence"]
    assert evidence[0]["tool_call_id"] == "call_1"
    assert evidence[0]["result"] == {"success": True, "value": 9}
    assert agent.trace.steps[0].plan_update["action"] == "complete_current_step"
    assert agent.state.plan is not None
    assert agent.state.plan.status == "completed"


def test_final_answer_guard_rejects_an_answer_until_dynamic_plan_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = install_responses(
        monkeypatch,
        FakeMessage(tool_calls=[make_tool_call(arguments={"value": 1})]),
        FakeMessage(content="Premature answer."),
        FakeMessage(
            tool_calls=[
                make_tool_call(arguments={"value": 2}, call_id="call_2")
            ]
        ),
        FakeMessage(content="Final verified answer."),
    )
    evaluator = FakePlanEvaluator(
        PlanUpdate.from_dict(
            {"action": "keep_current_step", "reason": "More evidence is needed."}
        ),
        PlanUpdate.from_dict(
            {"action": "complete_current_step", "result": "Evidence is complete."}
        ),
    )
    agent = Agent(
        task=make_task(),
        planner=FakePlanner(),
        plan_evaluator=evaluator,
        final_answer_policy=FinalAnswerPolicy(),
        config=AgentConfig(
            planning_mode=PlanningMode.DYNAMIC,
            enable_final_answer_guard=True,
        ),
    )

    answer = agent.run("Verify the result.", max_steps=4)

    assert answer == "Final verified answer."
    assert agent.trace.steps[1].final_answer_decision["allowed"] is False
    assert agent.trace.steps[3].final_answer_decision["allowed"] is True
    third_request_messages, _ = requests[2]
    assert any(
        message.get("role") == "system"
        and "not accepted as the final answer" in message.get("content", "")
        for message in third_request_messages
    )


def test_tool_budget_stops_before_executing_excess_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_responses(
        monkeypatch,
        FakeMessage(
            tool_calls=[
                make_tool_call(arguments={"value": 1}, call_id="call_1"),
                make_tool_call(arguments={"value": 2}, call_id="call_2"),
            ]
        ),
    )
    agent = Agent(task=make_task(), config=AgentConfig(planning_mode=PlanningMode.NONE))

    answer = agent.run("Store values.", max_tool_calls=1)

    assert "maximum number of allowed tool calls" in answer
    assert agent.state.status is RunStatus.FAILED
    assert agent.last_outcome is not None
    assert agent.last_outcome.stop_reason == "tool_budget_exceeded"
    assert agent.state.task_state["values"] == [1]
    assert agent.trace.tool_calls_used == 1


@pytest.mark.parametrize(
    ("run_kwargs", "message", "stop_reason"),
    [
        ({"max_steps": 0}, "max_steps must be greater than zero.", "invalid_max_steps"),
        (
            {"max_tool_calls": 0},
            "max_tool_calls must be greater than zero.",
            "invalid_max_tool_calls",
        ),
    ],
)
def test_run_rejects_invalid_budgets(run_kwargs, message, stop_reason) -> None:
    agent = Agent(task=make_task(), config=AgentConfig(planning_mode=PlanningMode.NONE))

    assert agent.run("Do work.", **run_kwargs) == message
    assert agent.state.status is RunStatus.FAILED
    assert agent.last_outcome is not None
    assert agent.last_outcome.stop_reason == stop_reason
    assert agent.trace.finished_at is not None


def test_run_records_llm_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_chat(messages, tools):
        raise LLMClientError("simulated connection failure")

    monkeypatch.setattr(agent_module, "chat", fail_chat)
    agent = Agent(task=make_task(), config=AgentConfig(planning_mode=PlanningMode.NONE))

    answer = agent.run("Do work.")

    assert "simulated connection failure" in answer
    assert agent.state.status is RunStatus.FAILED
    assert agent.last_outcome is not None
    assert agent.last_outcome.stop_reason == "llm_error"
    assert agent.trace.steps[0].error == answer


def test_each_run_gets_fresh_state_and_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    install_responses(
        monkeypatch,
        FakeMessage(tool_calls=[make_tool_call(arguments={"value": 1})]),
        FakeMessage(content="First run."),
        FakeMessage(content="Second run."),
    )
    agent = Agent(task=make_task(), config=AgentConfig(planning_mode=PlanningMode.NONE))

    assert agent.run("First.", max_steps=2, task_input={"label": "one"}) == "First run."
    first_trace = agent.trace
    assert agent.state.task_state["values"] == [1]

    assert agent.run("Second.", task_input={"label": "two"}) == "Second run."
    assert agent.trace is not first_trace
    assert agent.trace.tool_calls_used == 0
    assert agent.state.task_state == {"label": "two", "values": []}


@pytest.mark.parametrize(
    ("tool_call", "expected_error"),
    [
        (make_tool_call(arguments="{bad json"), "Invalid JSON arguments"),
        (make_tool_call(arguments=[]), "must be a JSON object"),
        (make_tool_call(name="missing_tool"), "Tool does not exist"),
        (make_tool_call(arguments={}), "Tool execution failed"),
    ],
)
def test_execute_tool_normalizes_invalid_calls(tool_call, expected_error) -> None:
    agent = Agent(task=make_task(), config=AgentConfig(planning_mode=PlanningMode.NONE))
    agent._start_run(
        user_input="Test a tool call.",
        task_input={},
        max_steps=1,
        max_tool_calls=1,
    )

    result, trace = agent.execute_tool(tool_call)

    assert result["success"] is False
    assert expected_error in result["error"]
    assert trace.success is False
    assert agent.state.errors[-1] == result["error"]


@pytest.mark.parametrize(
    ("tool_result", "expected_error"),
    [
        ("not an object", "expected an object"),
        ({"value": 1}, "missing boolean success"),
    ],
)
def test_execute_tool_rejects_invalid_results(tool_result, expected_error) -> None:
    def invalid_tool(value: int):
        return tool_result

    agent = Agent(
        task=make_task(function=invalid_tool),
        config=AgentConfig(planning_mode=PlanningMode.NONE),
    )
    agent._start_run(
        user_input="Test a tool result.",
        task_input={},
        max_steps=1,
        max_tool_calls=1,
    )

    result, _ = agent.execute_tool(make_tool_call())

    assert result["success"] is False
    assert expected_error in result["error"]


def test_execute_tool_converts_exceptions_to_failures() -> None:
    def failing_tool(value: int) -> dict[str, Any]:
        raise RuntimeError("boom")

    agent = Agent(
        task=make_task(function=failing_tool),
        config=AgentConfig(planning_mode=PlanningMode.NONE),
    )
    agent._start_run(
        user_input="Test a tool failure.",
        task_input={},
        max_steps=1,
        max_tool_calls=1,
    )

    result, _ = agent.execute_tool(make_tool_call())

    assert result == {"success": False, "error": "Tool execution failed: boom"}


def test_execute_tool_blocks_a_third_identical_attempt() -> None:
    agent = Agent(task=make_task(), config=AgentConfig(planning_mode=PlanningMode.NONE))
    agent._start_run(
        user_input="Test repeated calls.",
        task_input={},
        max_steps=1,
        max_tool_calls=3,
    )
    tool_call = make_tool_call(arguments={"value": 3})

    first_result, _ = agent.execute_tool(tool_call)
    second_result, _ = agent.execute_tool(tool_call)
    third_result, _ = agent.execute_tool(tool_call)

    assert first_result["success"] is True
    assert second_result["success"] is True
    assert third_result == {
        "success": False,
        "error": "Repeated tool call detected. Try another approach.",
    }
