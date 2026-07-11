from agent.agent import Agent

agent = Agent()

answer = agent.run(
  """
  Analyze this Github repository:
  https://github.com/wukaihang10/my_demo  
""", 5
)

print(answer)