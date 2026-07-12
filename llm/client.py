import os

from dotenv import load_dotenv

load_dotenv()

_client = None


def get_client():
  global _client

  if _client is not None:
    return _client

  try:
    from openai import OpenAI
  except ImportError as error:
    raise RuntimeError(
      "LLM dependencies are missing; install openai and python-dotenv"
    ) from error


  api_key = os.getenv("DEEPSEEK_API_KEY")
  base_url = os.getenv("DEEPSEEK_BASE_URL")

  if not api_key:
    raise RuntimeError("DEEPSEEK_API_KEY is not configured")
  if not base_url:
    raise RuntimeError("DEEPSEEK_BASE_URL is not configured")

  _client = OpenAI(
    api_key=api_key,
    base_url=base_url,
  )

  return _client

def chat(messages, tools):
  model = os.getenv("MODEL_NAME")
  if not model:
    raise RuntimeError("MODEL_NAME is not configured")

  response = get_client().chat.completions.create(
    model=model,
    messages=messages,
    tools=tools,
  )

  return response.choices[0].message
