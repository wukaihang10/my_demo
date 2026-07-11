from agent.agent import Agent


def analyze_repository(repo_url: str) -> str:
    agent = Agent()

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
   - What the project can currently do

7. Running instructions
   - How to install dependencies
   - How to start or test the project

8. Uncertainties
   - Clearly state anything that could not be confirmed

Requirements:
- Use repository tools to inspect the actual files.
- Read the README if it exists.
- Inspect the repository structure.
- Read the most relevant source and configuration files.
- Use search_code when an implementation location is unclear.
- Do not guess.
- Do not claim something unless repository evidence supports it.
"""

    return agent.run(task)