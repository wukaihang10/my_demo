import json

import pytest

import agent.observation as observation_module
from agent.observation import process_observation
from agent.outcome import AgentRunOutcome
from agent.run_context import RunContext
from agent.state import AgentState
from agent.tool_history import ToolHistory
from agent.trace import AgentTrace, StepTrace
from llm.messages import serialize_tool_call, tool_result_message


class DumpableToolCall:
    def model_dump(self, exclude_none: bool = False):
        return {"id": "call_1", "type": "function"}


def make_run_context() -> RunContext[dict]:
    return RunContext(
        user_input="Do work.",
        task_input={"label": "demo"},
        state=AgentState(task_state={}),
        trace=AgentTrace(max_steps=2, max_tool_calls=3),
        messages=[],
        tool_history=ToolHistory(),
    )


def test_run_context_is_finished_only_after_outcome_is_set() -> None:
    context = make_run_context()

    assert context.is_finished is False
    context.outcome = AgentRunOutcome.completed(answer="Done.")
    assert context.is_finished is True


def test_trace_serializes_usage_budgets_and_finish_time() -> None:
    trace = AgentTrace(max_steps=2, max_tool_calls=3)
    trace.add_step(StepTrace(step=1))
    trace.record_tool_call()
    trace.finish()

    data = trace.to_dict()

    assert data["steps_used"] == 1
    assert data["steps_remaining"] == 1
    assert data["tool_calls_used"] == 1
    assert data["tool_calls_remaining"] == 2
    assert data["finished_at"] is not None
    assert "status" not in data


def test_tool_history_detects_calls_at_the_configured_threshold() -> None:
    history = ToolHistory()
    arguments = {"value": 1}

    history.add("store", arguments)
    assert history.repeated("store", arguments) is False

    history.add("store", arguments)
    assert history.repeated("store", arguments) is True
    assert history.repeated("store", arguments, threshold=3) is False


def test_process_observation_wraps_oversized_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(observation_module, "MAX_TOOL_RESULT_CHARS", 120)

    encoded = process_observation({"success": True, "content": "x" * 500})
    payload = json.loads(encoded)

    assert len(encoded) <= 120
    assert payload["success"] is True
    assert payload["truncated"] is True
    assert len(payload["result_preview"]) < 500


def test_tool_result_message_encodes_a_json_object() -> None:
    message = tool_result_message(
        tool_call_id="call_1",
        tool_name="store_value",
        result={"success": True, "value": 4},
    )

    assert message["role"] == "tool"
    assert json.loads(message["content"]) == {"success": True, "value": 4}


def test_serialize_tool_call_supports_dicts_and_model_objects() -> None:
    raw = {"id": "call_1", "type": "function"}

    assert serialize_tool_call(raw) == raw
    assert serialize_tool_call(raw) is not raw
    assert serialize_tool_call(DumpableToolCall()) == raw

    with pytest.raises(TypeError, match="dictionary or support model_dump"):
        serialize_tool_call(object())
