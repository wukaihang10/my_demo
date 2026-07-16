from agent.final_answer import FinalAnswerPolicy
from agent.plan import AgentPlan, PlanStepSpec


def make_running_plan() -> AgentPlan:
    plan = AgentPlan.from_step_specs(
        goal="Complete the task.",
        step_specs=[
            PlanStepSpec(
                description="Collect the required evidence.",
                completion_criteria=("Enough evidence has been collected."),
            )
        ],
    )

    plan.start()
    return plan


def make_completed_plan() -> AgentPlan:
    plan = make_running_plan()
    plan.complete_current_step("The evidence was collected.")
    return plan


def test_policy_rejects_empty_answer() -> None:
    policy = FinalAnswerPolicy()

    decision = policy.evaluate(
        plan=None,
        tool_calls_used=0,
        response_content="   ",
    )

    assert decision.allowed is False
    assert "empty" in decision.reason.lower()


def test_policy_allows_answer_without_plan() -> None:
    policy = FinalAnswerPolicy()

    decision = policy.evaluate(
        plan=None,
        tool_calls_used=0,
        response_content="The answer is 42.",
    )

    assert decision.allowed is True


def test_policy_allows_direct_answer_before_tools() -> None:
    policy = FinalAnswerPolicy()
    plan = make_running_plan()

    decision = policy.evaluate(
        plan=plan,
        tool_calls_used=0,
        response_content="This task can be answered directly.",
    )

    assert decision.allowed is True


def test_policy_can_disable_direct_answers() -> None:
    policy = FinalAnswerPolicy(
        allow_direct_answer_before_tool_use=False,
    )
    plan = make_running_plan()

    decision = policy.evaluate(
        plan=plan,
        tool_calls_used=0,
        response_content="A proposed answer.",
    )

    assert decision.allowed is False


def test_policy_rejects_answer_during_execution() -> None:
    policy = FinalAnswerPolicy()
    plan = make_running_plan()

    decision = policy.evaluate(
        plan=plan,
        tool_calls_used=1,
        response_content="A premature final answer.",
    )

    assert decision.allowed is False
    assert "still in progress" in decision.reason


def test_policy_allows_answer_for_completed_plan() -> None:
    policy = FinalAnswerPolicy()
    plan = make_completed_plan()

    decision = policy.evaluate(
        plan=plan,
        tool_calls_used=2,
        response_content="The completed result.",
    )

    assert decision.allowed is True


def test_policy_rejects_answer_for_failed_plan() -> None:
    policy = FinalAnswerPolicy()
    plan = make_running_plan()

    plan.fail_current_step("The required resource is unavailable.")

    decision = policy.evaluate(
        plan=plan,
        tool_calls_used=1,
        response_content="A normal final answer.",
    )

    assert decision.allowed is False
    assert "failed" in decision.reason.lower()
