from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StagnationSnapshot:
    current_step_id: int | None
    attempts_on_current_step: int
    consecutive_keeps: int
    consecutive_evaluation_errors: int
    consecutive_rejected_final_answers: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StagnationDecision:
    should_stop: bool
    stop_reason: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StagnationPolicy:
    max_attempts_per_step: int = 5
    max_consecutive_keeps: int = 3
    max_consecutive_evaluation_errors: int = 2
    max_consecutive_rejected_final_answers: int = 2

    def __post_init__(self) -> None:
        limits = {
            "max_attempts_per_step": self.max_attempts_per_step,
            "max_consecutive_keeps": self.max_consecutive_keeps,
            "max_consecutive_evaluation_errors": self.max_consecutive_evaluation_errors,
            "max_consecutive_rejected_final_answer": self.max_consecutive_rejected_final_answers,
        }

        for name, value in limits.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero.")

    def evaluate(
        self,
        snapshot: StagnationSnapshot,
    ) -> StagnationDecision:
        if (
            snapshot.consecutive_evaluation_errors
            >= self.max_consecutive_evaluation_errors
        ):
            return StagnationDecision(
                should_stop=True,
                stop_reason="stagnation_evaluation_errors",
                message=f"Plan progress evaluation repeatedly failed {snapshot.consecutive_evaluation_errors} times.",
            )

        if (
            snapshot.consecutive_rejected_final_answers
            >= self.max_consecutive_rejected_final_answers
        ):
            return StagnationDecision(
                should_stop=True,
                stop_reason="stagnation_rejected_answers",
                message="The executor repeatedly proposed a final answer while the task plan was still incomplete.",
            )

        if snapshot.consecutive_keeps >= self.max_consecutive_keeps:
            return StagnationDecision(
                should_stop=True,
                stop_reason="stagnation_consecutive_keeps",
                message="The current plan step repeatedly remained in progress without sufficient evidence of completion.",
            )

        if snapshot.attempts_on_current_step >= self.max_attempts_per_step:
            return StagnationDecision(
                should_stop=True,
                stop_reason="stagnation_step_attempts",
                message=f"The current plan step exceeded the maximum number of execution attempts ({self.max_attempts_per_step}).",
            )

        return StagnationDecision(
            should_stop=False,
        )


class StagnationTracker:
    def __init__(self) -> None:
        self.current_step_id: int | None = None
        self.attempts_on_current_step = 0
        self.consecutive_keeps = 0
        self.consecutive_evaluation_errors = 0
        self.consecutive_rejected_final_answers = 0

    def reset(self) -> None:
        self.current_step_id = None
        self.attempts_on_current_step = 0
        self.consecutive_keeps = 0
        self.consecutive_evaluation_errors = 0
        self.consecutive_rejected_final_answers = 0

    def record_tool_batch(
        self,
        step_id: int | None,
    ) -> None:
        self._ensure_step(step_id)

        if step_id is not None:
            self.attempts_on_current_step += 1

        self.consecutive_rejected_final_answers = 0

    def record_plan_update(
        self,
        *,
        action: str,
        current_step_id: int | None,
    ) -> None:
        if action == "keep_current_step":
            self._ensure_step(current_step_id)
            self.consecutive_keeps += 1
            self.consecutive_evaluation_errors = 0
            return

        if action == "append_steps":
            self._ensure_step(current_step_id)

            # The plan changed, so this is progress at the plan
            # level, even though the current step did not change.
            self.consecutive_keeps = 0
            self.consecutive_evaluation_errors = 0
            self.consecutive_rejected_final_answers = 0
            return

        if action in {
            "complete_current_step",
            "skip_current_step",
            "fail_current_step",
        }:
            self._move_to_step(current_step_id)
            return

        raise ValueError(f"Unsupported plan update action: {action!r}.")

    def record_evaluation_error(
        self,
        step_id: int | None,
    ) -> None:
        self._ensure_step(step_id)
        self.consecutive_evaluation_errors += 1

    def record_final_answer_rejection(
        self,
        step_id: int | None,
    ) -> None:
        self._ensure_step(step_id)
        self.consecutive_rejected_final_answers += 1

    def snapshot(self) -> StagnationSnapshot:
        return StagnationSnapshot(
            current_step_id=self.current_step_id,
            attempts_on_current_step=self.attempts_on_current_step,
            consecutive_keeps=self.consecutive_keeps,
            consecutive_evaluation_errors=self.consecutive_evaluation_errors,
            consecutive_rejected_final_answers=self.consecutive_rejected_final_answers,
        )

    def _ensure_step(
        self,
        step_id: int | None,
    ) -> None:
        if step_id != self.current_step_id:
            self._move_to_step(step_id)

    def _move_to_step(
        self,
        step_id: int | None,
    ) -> None:
        self.reset()

        self.current_step_id = step_id
