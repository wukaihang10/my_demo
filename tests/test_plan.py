from agent.plan import build_repository_analysis_plan


def test_plan_finish_completes_final_step() -> None:
  plan = build_repository_analysis_plan("Analyze the repository.")
  plan.start()

  plan.finish("Repository analysis completed.")

  assert plan.status == "completed"
  assert plan.current_step is None
  assert plan.current_step_index == len(plan.steps)

  for step in plan.steps[:-1]:
    assert step.status == "skipped"

  final_step = plan.steps[-1]

  assert final_step.status == "completed"
  assert final_step.result == ("Repository analysis completed.")


def test_plan_finish_preserves_completed_steps() -> None:
  plan = build_repository_analysis_plan("Analyze the repository.")
  plan.start()

  plan.complete_current_step("Repository is available locally.")

  plan.finish("Repository analysis completed.")

  assert plan.steps[0].status == "completed"
  assert plan.steps[0].result == ("Repository is available locally.")

  for step in plan.steps[1:-1]:
    assert step.status == "skipped"

  assert plan.steps[-1].status == "completed"
  assert plan.status == "completed"


def test_plan_fail_marks_current_step_failed() -> None:
  plan = build_repository_analysis_plan("Analyze the repository.")
  plan.start()

  plan.fail("The language model request failed.")

  assert plan.status == "failed"
  assert plan.current_step is not None
  assert plan.current_step.status == "failed"
  assert plan.current_step.error == "The language model request failed."


def test_plan_fail_is_idempotent() -> None:
  plan = build_repository_analysis_plan("Analyze the repository.")
  plan.start()

  plan.fail("First failure.")
  plan.fail("Second failure.")

  assert plan.status == "failed"
  assert plan.current_step is not None
  assert plan.current_step.error == "First failure."
