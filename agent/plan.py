from dataclasses import asdict, dataclass, field
from typing import Any, Literal

PlanStepStatus = Literal[
  "pending",
  "in_progress",
  "completed",
  "failed",
  "skipped",
]

AgentPlanStatus = Literal[
  "pending",
  "in_progress",
  "completed",
  "failed",
]

@dataclass
class PlanStep:
  id: int 
  description: str
  expected_evidence: str | None = None

  status: PlanStepStatus = "pending"
  result: str | None = None
  error: str | None = None

  def start(self) -> None:
    if self.status == "in_progress":
      return
    
    if self.status != "pending":
      raise ValueError(f"Cannot start plan step {self.id} from status '{self.status}'.")
    
    self.status = "in_progress"

  
  def complete(self, result: str | None = None) -> None:
    if self.status not in ("pending", "in_progress"):
      raise ValueError(f"Cannot complete plan step {self.id} from satus '{self.status}'.")
    
    self.status = "completed"
    self.result = result
    self.error = None

  def fail(self, error: str) -> None:
    if self.status not in ("pending", "in_progress"):
      raise ValueError(f"Cannot fail plan step {self.id} from status '{self.status}'.")
    
    self.status = "failed"
    self.error = error

  
  def skip(self, reason: str | None = None) -> None:
    if self.status not in ("pending", "in_progress"):
      raise ValueError(f"Cannot skip plan step {self.id} from status '{self.status}'.")
    
    self.status = "skipped"
    self.result = reason
    self.error = None


@dataclass
class AgentPlan:
  goal: str
  steps: list[PlanStep] = field(default_factory = list)

  current_step_index: int = 0
  status: AgentPlanStatus = "pending"

  @property
  def current_step(self) -> PlanStep | None:
    if self.current_step_index >= len(self.steps):
      return None
    
    return self.steps[self.current_step_index]
  
  
  def start(self) -> None:
    if self.status == "in_progress":
      return
    
    if self.status != "pending":
      raise ValueError(f"Cannot start plan from status '{self.status}'.")
    
    if not self.steps:
      self.status = "completed"
      return 
    
    self.status = "in_progress"
    self.steps[self.current_step_index].start()

  
  def complete_current_step(
      self,
      result: str | None = None,
  ) -> None:
    step = self.current_step

    if step is None:
      raise ValueError("The plan has no current step.")
    
    step.complete(result)
    self._move_to_next_step()

  
  def fail_current_step(self, error: str) -> None:
    step = self.current_step

    if step is None:
      raise ValueError("The plan has no current step.")
    
    step.fail(error)

  
  def skip_current_step(
      self,
      reason: str | None = None, 
  ) -> None:
    step = self.current_step

    if step is None:
      raise ValueError("The plan has no current step.")
    
    step.skip(reason)
    self._move_to_next_step()

  
  def finish(
      self,
      result: str | None = None,
  ) -> None:
    if self.status == "completed":
      return 
    
    if self.status == "failed":
      raise ValueError("Cannot finish a failed plan.")
    
    if not self.steps:
      self.current_step_index = 0
      self.status = "completed"
      return
    
    final_step_index = len(self.steps) - 1

    for index, step in enumerate(self.steps):
      if step.status in ("completed", "failed", "skipped",):
        continue

      if index == final_step_index:
        step.complete(result)
      else:
        step.skip("Agent finished before this step was explicitely complete")

    self.current_step_index = len(self.steps)
    self.status = "completed"


  def fail(
    self,
    error: str,
  ) -> None:
    if self.status == "failed":
      return 
    
    if self.status == "completed":
      raise ValueError("Cannot fail a completed plan.")
    
    step = self.current_step

    if step is not None and step.status in ("pending", "in_progress",):
      step.fail(error)

    self.status = "failed"


  def _move_to_next_step(self) -> None:
    self.current_step_index += 1

    if self.current_step_index >= len(self.steps):
      self.status = "completed"
      return
    
    self.steps[self.current_step_index].start()

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)
  

def build_repository_analysis_plan(goal: str) -> AgentPlan:
  return AgentPlan(
    goal = goal,
    steps = [
      PlanStep(
        id = 1,
        description = "Prepare the repository for inspection.",
        expected_evidence = "A valid local path to the target repository."
      ),
      PlanStep(
        id = 2,
        description = "Collect a high-level overview of the repository.",
        expected_evidence = "Repository structure, languages, file counts and important files."
      ),
      PlanStep(
        id = 3,
        description = "Inspect the README and important configuration files.",
        expected_evidence = "The repository purpose, setup instructions, dependencies and enrty points."
      ),
      PlanStep(
        id = 4,
        description = "Inspect source files relevant to the user's request.",
        expected_evidence = "Concrete implementation details from relevant files.",
      ),
      PlanStep(
        id = 5,
        description = "Synthesize the evidence and produce the final answer.",
        expected_evidence = "An answer supported by the inspected repository feils."
      )
    ]
  )
  
  
    