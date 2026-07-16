from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StagnationSnapshot:
    current_step_id: int | None
    attempts_on_current_step: int
    consecutive_keeps: int
    consecutive_evaluation_errors: int
    consecutive_rejected_final_answers: int
    recovery_attempts_on_current_step: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StagnationDecision:
    should_stop: bool = False
    should_recover: bool = False
    trigger: str | None = None
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

    # 设为 0 可以关闭恢复机制，保持以前“触发即停止”的行为。
    max_recovery_attempts_per_step: int = 1

    def __post_init__(self) -> None:
        positive_limits = {
            "max_attempts_per_step": self.max_attempts_per_step,
            "max_consecutive_keeps": self.max_consecutive_keeps,
            "max_consecutive_evaluation_errors": (
                self.max_consecutive_evaluation_errors
            ),
            "max_consecutive_rejected_final_answers": (
                self.max_consecutive_rejected_final_answers
            ),
        }

        for name, value in positive_limits.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero.")

        if self.max_recovery_attempts_per_step < 0:
            raise ValueError("max_recovery_attempts_per_step must not " "be negative.")

    def evaluate(
        self,
        snapshot: StagnationSnapshot,
    ) -> StagnationDecision:
        trigger = self._detect_trigger(snapshot)

        if trigger is None:
            return StagnationDecision()

        stop_reason, message = trigger

        # 评估器连续失败不是 Executor 换工具或换方法能够
        # 修复的问题，因此不进入 Executor 恢复流程。
        if stop_reason == "stagnation_evaluation_errors":
            return StagnationDecision(
                should_stop=True,
                trigger=stop_reason,
                stop_reason=stop_reason,
                message=message,
            )

        recovery_available = (
            snapshot.recovery_attempts_on_current_step
            < self.max_recovery_attempts_per_step
        )

        if recovery_available:
            return StagnationDecision(
                should_recover=True,
                trigger=stop_reason,
                message=message,
            )

        return StagnationDecision(
            should_stop=True,
            trigger=stop_reason,
            stop_reason=stop_reason,
            message=(
                f"{message} The recovery allowance for the "
                "current plan step has been exhausted."
            ),
        )

    def _detect_trigger(
        self,
        snapshot: StagnationSnapshot,
    ) -> tuple[str, str] | None:
        if (
            snapshot.consecutive_evaluation_errors
            >= self.max_consecutive_evaluation_errors
        ):
            return (
                "stagnation_evaluation_errors",
                (
                    "Plan progress evaluation repeatedly failed "
                    f"{snapshot.consecutive_evaluation_errors} "
                    "times."
                ),
            )

        if (
            snapshot.consecutive_rejected_final_answers
            >= self.max_consecutive_rejected_final_answers
        ):
            return (
                "stagnation_rejected_answers",
                (
                    "The executor repeatedly proposed a final "
                    "answer while the task plan was incomplete."
                ),
            )

        if snapshot.consecutive_keeps >= self.max_consecutive_keeps:
            return (
                "stagnation_consecutive_keeps",
                (
                    "The current plan step repeatedly remained "
                    "in progress without enough evidence of "
                    "completion."
                ),
            )

        if snapshot.attempts_on_current_step >= self.max_attempts_per_step:
            return (
                "stagnation_step_attempts",
                (
                    "The current plan step reached the maximum "
                    f"number of execution attempts "
                    f"({self.max_attempts_per_step})."
                ),
            )

        return None


class StagnationTracker:
    def __init__(self) -> None:
        self.current_step_id: int | None = None
        self.attempts_on_current_step = 0
        self.consecutive_keeps = 0
        self.consecutive_evaluation_errors = 0
        self.consecutive_rejected_final_answers = 0
        self.recovery_attempts_on_current_step = 0

    def reset(self) -> None:
        self.current_step_id = None
        self.attempts_on_current_step = 0
        self.consecutive_keeps = 0
        self.consecutive_evaluation_errors = 0
        self.consecutive_rejected_final_answers = 0
        self.recovery_attempts_on_current_step = 0

    def record_tool_batch(
        self,
        step_id: int | None,
    ) -> None:
        self._ensure_step(step_id)

        if step_id is not None:
            self.attempts_on_current_step += 1

        # 执行了工具，说明没有连续重复提交最终答案。
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

            # 计划发生了结构变化，视为计划级进展。
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
        self.consecutive_keeps = 0

    def record_final_answer_rejection(
        self,
        step_id: int | None,
    ) -> None:
        self._ensure_step(step_id)
        self.consecutive_rejected_final_answers += 1

    def record_recovery(
        self,
        step_id: int | None,
    ) -> None:
        if step_id is None:
            raise ValueError("A stagnation recovery requires a current " "plan step.")

        self._ensure_step(step_id)
        self.recovery_attempts_on_current_step += 1

        # 恢复之后给新的策略一个完整窗口。
        # recovery_attempts 不重置，只有进入新步骤才重置。
        self.attempts_on_current_step = 0
        self.consecutive_keeps = 0
        self.consecutive_evaluation_errors = 0
        self.consecutive_rejected_final_answers = 0

    def snapshot(self) -> StagnationSnapshot:
        return StagnationSnapshot(
            current_step_id=self.current_step_id,
            attempts_on_current_step=(self.attempts_on_current_step),
            consecutive_keeps=self.consecutive_keeps,
            consecutive_evaluation_errors=(self.consecutive_evaluation_errors),
            consecutive_rejected_final_answers=(
                self.consecutive_rejected_final_answers
            ),
            recovery_attempts_on_current_step=(self.recovery_attempts_on_current_step),
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
        self.current_step_id = step_id
        self.attempts_on_current_step = 0
        self.consecutive_keeps = 0
        self.consecutive_evaluation_errors = 0
        self.consecutive_rejected_final_answers = 0
        self.recovery_attempts_on_current_step = 0
