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
  error: str | None =None

@dataclass
class StepTrace:
  step: int
  tool_calls: list[ToolTrace] = field(default_factory=list)
  final_response: str | None = None

@dataclass
class AgentTrace:
  started_at: str = field(
    default_factory=lambda: datetime.now(timezone.utc).isoformat()
  )

  finished_at: str | None = None
  status: str = "running"
  steps: list[StepTrace] = field(default_factory=list)

  def add_step(self, step: StepTrace) -> None:
    self.steps.append(step)

  def finish(self, status: str) -> None:
    self.status = status
    self.finished_at = datetime.now(timezone.utc).isoformat()

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)