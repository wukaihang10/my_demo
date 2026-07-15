from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ToolTrace:
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    success: bool
    duration_ms: float
    result_preview: str
    error: str | None = None


@dataclass
class StepTrace:
    step: int
    tool_calls: list[ToolTrace] = field(default_factory=list)
    final_response: str | None = None
    error: str | None = (
        None  # 与ToolTrace里的error信息不同，上面的error是记录tool调用时出现的错误，step里的error是记录llm调用时出现的错误
    )
    plan_update: dict[str, Any] | None = None


@dataclass
class AgentTrace:
    max_steps: int | None = None
    max_tool_calls: int | None = None

    steps_used: int = 0
    tool_calls_used: int = 0

    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    finished_at: str | None = None
    status: str = "running"
    steps: list[StepTrace] = field(default_factory=list)

    def add_step(self, step: StepTrace) -> None:
        self.steps.append(step)
        self.steps_used += 1

    def record_tool_call(self) -> None:
        self.tool_calls_used += 1

    def finish(self, status: str) -> None:
        self.status = status
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        if self.max_steps is None:
            data["steps_remaining"] = None

        else:
            data["steps_remaining"] = max(self.max_steps - self.steps_used, 0)

        if self.max_tool_calls is None:
            data["tool_calls_remaining"] = None

        else:
            data["tool_calls_remaining"] = max(
                self.max_tool_calls - self.tool_calls_used, 0
            )

        return data
