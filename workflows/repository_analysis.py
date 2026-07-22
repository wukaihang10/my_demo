from typing import Any

from agent.agent import Agent
from tasks.repository import REPOSITORY_TASK
from agent.config import AgentConfig, PlanningMode


def analyze_repository(repo_url: str) -> dict[str, Any]:
    agent = Agent(
        task=REPOSITORY_TASK, config=AgentConfig(planning_mode=PlanningMode.STATIC)
    )

    task = f"""
Analyze the following GitHub repository:

{repo_url}


Your final answer must include:

1. Project overview
   - What problem the project solves
   - Who it is intended for

2. Technology stack
   - Main programming languages
   - Important libraries or frameworks
   - Build, package, or dependency tools

3. Repository structure
   - Important top-level directories and files
   - The purpose of each important directory

4. Entry points
   - Where the program starts
   - Important commands or scripts used to run it

5. Core implementation
   - Main modules, classes, and functions
   - How the important components work together

6. Current capabilities
   - What the project can currently do, what is this project's features compared to the other same projects.

7. Teach users how to use this repository.

8. Running instructions
   - How to install dependencies
   - How to start or test the project

9. Limitations and uncertainties
   - Missing files or incomplete implementations
   - Anything that could not be confirmed


Requirements:

- Do not guess.
- Do not claim anything unless repository evidence supports it.
- Response in Chinese.
- Clearly distinguish confirmed facts from reasonable inferences.
- If a tool fails, inspect the error and try a reasonable alternative.
"""

    answer = agent.run(
        user_input=task,
        max_steps=15,
        max_tool_calls=60,
        task_input={"repo_url": repo_url},
    )

    return {
        "answer": answer,
        "trace": agent.trace.to_dict(),
        "state": agent.state.to_dict(),
    }
