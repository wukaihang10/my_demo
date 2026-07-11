import subprocess
from pathlib import Path

WORKSPACE = Path("workspace/repos")

def clone_repository(url):

  WORKSPACE.mkdir(
    parents=True,
    exist_ok=True
  )

  repo_name = url.rstrip("/").split("/")[-1]

  target = WORKSPACE / repo_name

  if target.exists():
    return {
        "success": True,
        "repo_path": str(target),
        "message": "Repository already exists"
    }
  
  command = [
    "git",
    "clone",
    url,
    str(target)
  ]

  result = subprocess.run(
    command,
    capture_output = True,
    text = True
  )

  if result.returncode != 0:
    return {
    "success": False,
    "error": result.stderr
    }
  
  return {
    "success": True,

    "repo_path":
        str(target),

    "message":
        "Repository cloned successfully"
  }