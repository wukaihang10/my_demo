from dataclasses import dataclass
from enum import Enum


class PlanningMode(str, Enum):
    NONE = "none"
    STATIC = "static"
    DYNAMIC = "dynamic"


@dataclass(frozen=True)
class AgentConfig:
    planning_mode: PlanningMode = PlanningMode.STATIC

    # Final-answer evaluation is an additional LLM/policy layer.
    # Keep it disabled by default until evals prove its value.
    enable_final_answer_guard: bool = False
    enable_stagnation_recovery: bool = False
