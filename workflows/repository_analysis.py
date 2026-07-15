from agent.agent import Agent
from typing import Any


def analyze_repository(repo_url: str) -> dict[str, Any]:
    agent = Agent()

    task = f"""
Analyze the following GitHub repository:

{repo_url}

Follow this process:

1. Clone the repository if it is not already available locally.
2. Call summarize_repository to obtain a high-level overview.
3. Use list_files only when you need a more detailed repository tree.
4. Read the README if it exists.
5. Read the most relevant source and configuration files.
6. Use search_code when the location of an implementation is unclear.
7. Do not read every file unless the repository is very small.
8. Base every conclusion on repository evidence.

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
   - What the project can currently do

7. Running instructions
   - How to install dependencies
   - How to start or test the project

8. Limitations and uncertainties
   - Missing files or incomplete implementations
   - Anything that could not be confirmed

Requirements:

- Do not guess.
- Do not claim anything unless repository evidence supports it.
- Clearly distinguish confirmed facts from reasonable inferences.
- If a tool fails, inspect the error and try a reasonable alternative.
"""

    answer = agent.run(task, max_steps=15, repo_url=repo_url)

    return {
        "answer": answer,
        "trace": agent.trace.to_dict(),
        "state": agent.state.to_dict(),
    }
