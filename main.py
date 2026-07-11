from agent.agent import Agent


agent = Agent()

result = agent.run(
    """
Analyze this GitHub repository:

https://github.com/wukaihang10/my_demo

Explain:
1. What the project does
2. Its architecture
3. Its main execution flow
4. The purpose of each important module
"""
)

print(result)