from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
  api_key = os.getenv("DEEPSEEK_API_KEY"),
  base_url = os.getenv("DEEPSEEK_BASE_URL")
)

def chat(messages, tools):
  response = client.chat.completions.create(
    model = os.getenv("MODEL_NAME"),
    messages = messages,
    tools = tools
  )

  return response.choices[0].message