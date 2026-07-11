from pathlib import Path

def list_files(repo_path):
  path = Path(repo_path)

  if not path.exists():
    return {
      "success": False,
      "error": f"Path does not exist: {repo_path}" 
    }
  
  files = []

  for item in path.rglob("*"):
    if item.is_file():
      # relative = item.relative_to(path)
      files.append(str(item))

  return {
    "success": True,
    "content": "\n".join(files)
  }


def read_file(file_path):

  path = Path(file_path)

  if not path.exists():
    return {
      "success": False,

      "error":
        f"File not found: {file_path}"
    }
  
  try:
    content = path.read_text(encoding = "utf-8")

    return {
      "success": True,
      "content": content
    }
  except Exception as e:
    return {
      "success": False,
      "error": str(e)
    }
  

def search_code(repo_path, keyword):
  root = Path(repo_path)

  if not root.exists():
    return {
      "success": False,
      "errot": f"Path not found: {repo_path}"
    }
  
  matches = []

  for file in root.rglob("*"):
    if not file.is_file():
      continue

    if any(
      part in [
        ".git",
        "node_modules",
        "__pycache__"
      ]
      for part in file.parts
    ):
      continue

    try:
      content = file.read_text(encoding="utf-8")

      if keyword.lower() in content.lower():
        matches.append(str(file))

    except Exception:
      continue

  return {
    "success": True,
    "keyword": keyword,
    "matches": matches
  }