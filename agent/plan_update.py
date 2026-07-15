from dataclasses import dataclass
from typing import Any, Literal, cast

from agent.plan import (
    AgentPlan,
    PlanError,
    PlanStepSpec,
    PlanValidationError,
)

PlanUpdateAction = Literal[
    "keep_current_step",
    "complete_current_step",
    "skip_current_step",
    "fail_current_step",
    "append_steps",
]

VALID_PLAN_UPDATE_ACTIONS = {
    "keep_current_step",
    "complete_current_step",
    "skip_current_step",
    "fail_current_step",
    "append_steps",
}


class PlanUpdateError(ValueError):
    """Base exception for controlled Plan updates."""


class PlanUpdateValidationError(PlanUpdateError):
    """Raised when a requested plan updates has an invalid structure."""


class PlanUpdateBudgetError(PlanUpdateError):
    """Raised when a plan update exceeds an update budget."""


class PlanUpdateApplicationError(PlanUpdateError):
    """Raised when a validated update cannot be applied to the plan."""


@dataclass(frozen=True)
class PlanUpdate:
    """
    A declarative request to update a running plan.
    The LLM may propose this object, but only PlanController may apply it.
    """

    action: PlanUpdateAction

    result: str | None = None
    reason: str | None = None
    error: str | None = None

    new_steps: tuple[PlanStepSpec, ...] = ()

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
    ) -> "PlanUpdate":
        if not isinstance(payload, dict):
            raise PlanUpdateValidationError("Plan update payload must be an object.")

        action = payload.get("action")

        if not isinstance(action, str):
            raise PlanUpdateValidationError(
                "Plan update field 'action' must be a string."
            )

        if action not in VALID_PLAN_UPDATE_ACTIONS:
            raise PlanUpdateValidationError(
                f"Unsupported plan update action: '{action}'."
            )

        typed_action = cast(PlanUpdateAction, action)

        if typed_action == "keep_current_step":
            cls._reject_unknown_fields(
                payload,
                allowed_fields={"action", "reason"},
            )
            return cls(
                action=typed_action,
                reason=cls._optional_text(
                    payload.get("reason"),
                    field_name="reason",
                ),
            )

        if typed_action == "complete_current_step":
            cls._reject_unknown_fields(
                payload,
                allowed_fields={"action", "result"},
            )

            return cls(
                action=typed_action,
                result=cls._optional_text(
                    payload.get("result"),
                    field_name="result",
                ),
            )

        if typed_action == "skip_current_step":
            cls._reject_unknown_fields(
                payload,
                allowed_fields={"action", "reason"},
            )

            return cls(
                action=typed_action,
                reason=cls._optional_text(
                    payload.get("reason"),
                    field_name="reason",
                ),
            )

        if typed_action == "fail_current_step":
            cls._reject_unknown_fields(
                payload,
                allowed_fields={"action", "error"},
            )

            return cls(
                action=typed_action,
                error=cls._required_text(
                    payload.get("error"),
                    field_name="error",
                ),
            )

        cls._reject_unknown_fields(
            payload,
            allowed_fields={"action", "steps"},
        )

        return cls(
            action=typed_action,
            new_steps=cls._parse_step_specs(payload.get("steps")),
        )

    def _reject_unknown_fields(
        payload: dict[str, Any],
        *,
        allowed_fields: set[str],
    ) -> None:
        unknown_fields = set(payload) - allowed_fields

        if unknown_fields:
            raise PlanUpdateValidationError(
                "Plan update contains unsupported fields: "
                + ", ".join(sorted(unknown_fields))
            )

    @staticmethod
    def _optional_text(
        value: Any,
        *,
        field_name: str,
    ) -> str | None:
        if value is None:
            return None

        if not isinstance(value, str):
            raise PlanUpdateValidationError(
                f"Plan update field '{field_name}' must be a string or null."
            )

        return value.strip() or None

    @staticmethod
    def _required_text(
        value: Any,
        *,
        field_name: str,
    ) -> str:
        if not isinstance(value, str):
            raise PlanUpdateValidationError(
                f"Plan update field '{field_name}' must be a non-empty string."
            )

        value = value.strip()

        if not value:
            raise PlanUpdateValidationError(
                f"Plan update field '{field_name} must not be empty.'"
            )

        return value

    @classmethod
    def _parse_step_specs(
        cls,
        raw_steps: Any,
    ) -> tuple[PlanStepSpec, ...]:
        if not isinstance(raw_steps, list):
            raise PlanUpdateValidationError("Plan update field 'steps' must be a list.")

        if not raw_steps:
            raise PlanUpdateValidationError(
                "Plan update must append at least one step."
            )

        allowed_fields = {
            "description",
            "completion_criteria",
        }

        step_specs: list[PlanStepSpec] = []

        for index, raw_step in enumerate(raw_steps, start=1):
            if not isinstance(raw_step, dict):
                raise PlanUpdateValidationError(
                    f"Appended plan step {index} must be an object."
                )

            unknown_fields = set(raw_step) - allowed_fields

            if unknown_fields:
                raise PlanUpdateValidationError(
                    f"Append plan step {index} contains unsupported fields: "
                    + ",".join(sorted(unknown_fields))
                )

            description = raw_step.get("description")
            completion_criteria = raw_step.get("completion_criteria")

            if not isinstance(description, str):
                raise PlanUpdateValidationError(
                    f"Appended plan step {index} field 'description' must be a string."
                )

            if completion_criteria is not None and not isinstance(
                completion_criteria, str
            ):
                raise PlanUpdateValidationError(
                    f"Appended plan step {index} field 'completion_criteria' must be a string or null."
                )

            try:
                step_spec = PlanStepSpec(
                    description=description,
                    completion_criteria=completion_criteria,
                )
            except PlanValidationError as error:
                raise PlanUpdateValidationError(
                    f"Appended plan step {index} is invalid: {error}"
                ) from error

            step_specs.append(step_spec)

        return tuple(step_specs)


@dataclass(frozen=True)
class PlanUpdatePolicy:
    """
    Program-owned limits for runtime plan updates.
    """

    max_updates: int = 12
    max_total_steps: int = 10
    max_added_steps_per_update: int = 3

    def __post_init__(self) -> None:
        if self.max_updates <= 0:
            raise ValueError("max_updates must be greater than zero.")

        if self.max_total_steps <= 0:
            raise ValueError("max_total_steps must be greater than zero.")

        if self.max_added_steps_per_update <= 0:
            raise ValueError("max_added_steps_per_update must be greater than zero.")


class PlanController:
    """
    Applies validated PlanUpdate objects to an AgentPlan.

    The controller owns update budgets. It does not allow callers to mutate arbitrary plan fields.
    """

    def __init__(
        self,
        policy: PlanUpdatePolicy | None = None,
    ) -> None:
        self.policy = policy or PlanUpdatePolicy()
        self.updates_used = 0

    @property
    def updates_remaining(self) -> int:
        return max(self.policy.max_updates - self.updates_used, 0)

    def apply_update(
        self,
        plan: AgentPlan,
        update: PlanUpdate,
    ) -> dict[str, Any]:
        plan_changed = update.action != "keep_current_step"

        if plan_changed and self.updates_used >= self.policy.max_updates:
            raise PlanUpdateBudgetError("The plan update budget has been exhausted.")

        added_step_ids: list[int] = []

        try:
            if update.action == "keep_current_step":
                if plan.status != "in_progress" or plan.current_step is None:
                    raise PlanUpdateApplicationError(
                        "The current step can only be kept while "
                        "the plan is in progress."
                    )

            elif update.action == "complete_current_step":
                plan.complete_current_step(
                    result=update.result,
                )

            elif update.action == "skip_current_step":
                plan.skip_current_step(
                    reason=update.reason,
                )

            elif update.action == "fail_current_step":
                if update.error is None:
                    raise PlanUpdateValidationError(
                        "fail_current_step requires an error."
                    )

                plan.fail_current_step(
                    error=update.error,
                )

            elif update.action == "append_steps":
                if not update.new_steps:
                    raise PlanUpdateValidationError(
                        "append_steps requires at least one step."
                    )

                if len(update.new_steps) > self.policy.max_added_steps_per_update:
                    raise PlanUpdateBudgetError(
                        "A single update may add at most "
                        f"{self.policy.max_added_steps_per_update} "
                        "steps."
                    )

                new_total = len(plan.steps) + len(update.new_steps)

                if new_total > self.policy.max_total_steps:
                    raise PlanUpdateBudgetError(
                        "The plan may contain at most "
                        f"{self.policy.max_total_steps} steps."
                    )

                added_steps = plan.append_step_specs(
                    update.new_steps,
                    max_steps=self.policy.max_total_steps,
                )

                added_step_ids = [step.id for step in added_steps]

            else:
                raise PlanUpdateValidationError(
                    "Unsupported plan update action: " f"'{update.action}'."
                )

        except PlanUpdateError:
            raise

        except PlanError as error:
            raise PlanUpdateApplicationError(
                f"Plan update could not be applied: {error}"
            ) from error

        if plan_changed:
            self.updates_used += 1

        current_step = plan.current_step

        return {
            "success": True,
            "action": update.action,
            "plan_changed": plan_changed,
            "plan_status": plan.status,
            "current_step_id": (current_step.id if current_step is not None else None),
            "added_step_ids": added_step_ids,
            "updates_used": self.updates_used,
            "updates_remaining": self.updates_remaining,
        }
