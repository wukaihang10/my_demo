import pytest

from agent.stagnation import (
    StagnationPolicy,
    StagnationTracker,
)


def test_policy_rejects_non_positive_limits() -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        StagnationPolicy(
            max_consecutive_keeps=0,
        )


def test_tracker_records_tool_attempts() -> None:
    tracker = StagnationTracker()

    tracker.record_tool_batch(step_id=1)
    tracker.record_tool_batch(step_id=1)

    snapshot = tracker.snapshot()

    assert snapshot.current_step_id == 1
    assert snapshot.attempts_on_current_step == 2


def test_tracker_resets_when_step_changes() -> None:
    tracker = StagnationTracker()

    tracker.record_tool_batch(step_id=1)
    tracker.record_plan_update(
        action="keep_current_step",
        current_step_id=1,
    )

    tracker.record_plan_update(
        action="complete_current_step",
        current_step_id=2,
    )

    snapshot = tracker.snapshot()

    assert snapshot.current_step_id == 2
    assert snapshot.attempts_on_current_step == 0
    assert snapshot.consecutive_keeps == 0
    assert snapshot.consecutive_evaluation_errors == 0


def test_keep_increments_keep_counter() -> None:
    tracker = StagnationTracker()

    tracker.record_tool_batch(step_id=1)
    tracker.record_plan_update(
        action="keep_current_step",
        current_step_id=1,
    )
    tracker.record_plan_update(
        action="keep_current_step",
        current_step_id=1,
    )

    snapshot = tracker.snapshot()

    assert snapshot.consecutive_keeps == 2


def test_append_steps_resets_keep_counter() -> None:
    tracker = StagnationTracker()

    tracker.record_tool_batch(step_id=1)
    tracker.record_plan_update(
        action="keep_current_step",
        current_step_id=1,
    )

    tracker.record_plan_update(
        action="append_steps",
        current_step_id=1,
    )

    snapshot = tracker.snapshot()

    assert snapshot.current_step_id == 1
    assert snapshot.attempts_on_current_step == 1
    assert snapshot.consecutive_keeps == 0


def test_policy_stops_after_consecutive_keeps() -> None:
    tracker = StagnationTracker()
    policy = StagnationPolicy(
        max_consecutive_keeps=2,
    )

    tracker.record_tool_batch(step_id=1)
    tracker.record_plan_update(
        action="keep_current_step",
        current_step_id=1,
    )

    first_decision = policy.evaluate(tracker.snapshot())

    assert first_decision.should_stop is False

    tracker.record_tool_batch(step_id=1)
    tracker.record_plan_update(
        action="keep_current_step",
        current_step_id=1,
    )

    second_decision = policy.evaluate(tracker.snapshot())

    assert second_decision.should_stop is True
    assert second_decision.stop_reason == ("stagnation_consecutive_keeps")


def test_policy_stops_after_evaluation_errors() -> None:
    tracker = StagnationTracker()
    policy = StagnationPolicy(
        max_consecutive_evaluation_errors=2,
    )

    tracker.record_evaluation_error(step_id=1)
    tracker.record_evaluation_error(step_id=1)

    decision = policy.evaluate(tracker.snapshot())

    assert decision.should_stop is True
    assert decision.stop_reason == ("stagnation_evaluation_errors")


def test_policy_stops_after_rejected_answers() -> None:
    tracker = StagnationTracker()
    policy = StagnationPolicy(
        max_consecutive_rejected_final_answers=2,
    )

    tracker.record_final_answer_rejection(step_id=1)
    tracker.record_final_answer_rejection(step_id=1)

    decision = policy.evaluate(tracker.snapshot())

    assert decision.should_stop is True
    assert decision.stop_reason == ("stagnation_rejected_answers")


def test_policy_stops_after_step_attempt_limit() -> None:
    tracker = StagnationTracker()
    policy = StagnationPolicy(
        max_attempts_per_step=2,
        max_consecutive_keeps=10,
    )

    tracker.record_tool_batch(step_id=1)
    tracker.record_tool_batch(step_id=1)

    decision = policy.evaluate(tracker.snapshot())

    assert decision.should_stop is True
    assert decision.stop_reason == ("stagnation_step_attempts")
