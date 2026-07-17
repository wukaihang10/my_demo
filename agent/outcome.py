from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AgentRunOutcome:
    """Internal result describing how one Agent run ended."""

    success: bool
    stop_reason: str
    answer: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stop_reason, str):
            raise TypeError("stop_reason must be a string.")

        if not self.stop_reason.strip():
            raise ValueError("stop_reason must not be empty.")

        if self.success:
            if not isinstance(self.answer, str):
                raise ValueError("A successful outcome requires an answer.")

            if not self.answer.strip():
                raise ValueError("A successfult outcome requires a non-empty answer.")

            if self.error is not None:
                raise ValueError("A successulf outcome must not contain an error.")

        else:
            if not isinstance(self.error, str):
                raise ValueError("A failed outcome requires an error.")

            if not self.error.strip():
                raise ValueError("A failed outcome requires a non-empty error.")

            if self.answer is not None:
                raise ValueError("A failed outcome must not contain an answer.")

    @property
    def response(self) -> str:
        if self.success:
            assert self.answer is not None
            return self.answer

        assert self.error is not None
        return self.error

    @classmethod
    def completed(
        cls,
        *,
        answer: str,
        stop_reason: str = "completed",
    ) -> "AgentRunOutcome":
        return cls(
            success=True,
            answer=answer,
            stop_reason=stop_reason,
        )

    @classmethod
    def failed(
        cls,
        *,
        error: str,
        stop_reason: str,
    ) -> "AgentRunOutcome":
        return cls(
            success=False,
            error=error,
            stop_reason=stop_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
