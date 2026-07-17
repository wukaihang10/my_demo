import pytest

from agent.outcome import AgentRunOutcome


def test_completed_outcome() -> None:
    outcome = AgentRunOutcome.completed(
        answer="Task completed.",
    )

    assert outcome.success is True
    assert outcome.answer == "Task completed."
    assert outcome.error is None
    assert outcome.stop_reason == "completed"
    assert outcome.response == "Task completed."


def test_failed_outcome() -> None:
    outcome = AgentRunOutcome.failed(
        error="Task failed.",
        stop_reason="tool_error",
    )

    assert outcome.success is False
    assert outcome.answer is None
    assert outcome.error == "Task failed."
    assert outcome.stop_reason == "tool_error"
    assert outcome.response == "Task failed."


def test_success_requires_answer() -> None:
    with pytest.raises(
        ValueError,
        match="requires an answer",
    ):
        AgentRunOutcome(
            success=True,
            stop_reason="completed",
        )


def test_failure_requires_error() -> None:
    with pytest.raises(
        ValueError,
        match="requires an error",
    ):
        AgentRunOutcome(
            success=False,
            stop_reason="failed",
        )


def test_success_cannot_contain_error() -> None:
    with pytest.raises(
        ValueError,
        match="must not contain an error",
    ):
        AgentRunOutcome(
            success=True,
            answer="Done.",
            error="Unexpected error.",
            stop_reason="completed",
        )


def test_failure_cannot_contain_answer() -> None:
    with pytest.raises(
        ValueError,
        match="must not contain an answer",
    ):
        AgentRunOutcome(
            success=False,
            answer="Done.",
            error="Failed.",
            stop_reason="failed",
        )


def test_stop_reason_must_not_be_empty() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        AgentRunOutcome.completed(
            answer="Done.",
            stop_reason="   ",
        )
