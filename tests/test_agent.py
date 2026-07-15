import json
from types import SimpleNamespace
from typing import Any

import agent.agent as agent_module
from agent.agent import Agent
from llm.client import LLMClientError
from agent.plan import AgentPlan, PlanStepSpec


class FakePlanner:
    def create_plan(self, goal: str) -> AgentPlan:
        return AgentPlan.from_step_specs(
            goal=goal,
            step_specs=[
                PlanStepSpec(
                    description="Execute the requested task.",
                    completion_criteria=("Enough evidence exists to answer the user."),
                )
            ],
        )


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

    agent = Agent(planner=FakePlanner())

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
    assert agent.state.plan.current_step is None

    assert all(step.status == "skipped" for step in agent.state.plan.steps[:-1])

    assert agent.state.plan.result == "Repository analysis completed."
    assert all(step.status == "skipped" for step in agent.state.plan.steps)


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

    agent = Agent(planner=FakePlanner())
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

    agent = Agent(planner=FakePlanner())
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

    agent = Agent(planner=FakePlanner())

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
