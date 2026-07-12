import json

MAX_TOOL_RESULT_CHARS = 12000

def process_observation(result: dict) -> str:
  text = json.dumps(result, ensure_ascii = False, default = str)

  if len(text) <= MAX_TOOL_RESULT_CHARS:
    return text

  preview = text[:MAX_TOOL_RESULT_CHARS]
  payload = {
    "success": result.get("success", False),
    "truncated": True,
    "result_preview": preview,
  }
  encoded = json.dumps(payload, ensure_ascii=False)

  while len(encoded) > MAX_TOOL_RESULT_CHARS and preview:
    overflow = len(encoded) - MAX_TOOL_RESULT_CHARS
    preview = preview[:max(0, len(preview) - overflow - 1)]
    payload["result_preview"] = preview
    encoded = json.dumps(payload, ensure_ascii=False)

  return encoded
