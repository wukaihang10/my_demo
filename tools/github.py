import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

WORKSPACE = Path(__file__).resolve().parent.parent / "workspace" / "repos"
GITHUB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")

def clone_repository(url: str):

  if not isinstance(url, str) or not url.strip():
    return {
      "success": False,
      "error": "Repository URL cannot be empty",
    }

  url = url.strip()
  parsed_url = urlparse(url)
  path_parts = [part for part in parsed_url.path.split("/") if part]

  if (
    parsed_url.scheme not in {"http", "https"}
    or parsed_url.hostname not in {"github.com", "www.github.com"}
    or len(path_parts) != 2
  ):
    return {
      "success": False,
      "error": "URL must identify a GitHub repository",
    }

  WORKSPACE.mkdir(
    parents=True,
    exist_ok=True
  )

  repo_name = path_parts[-1]
  if repo_name.endswith(".git"):
    repo_name = repo_name[:-4]

  owner_name = path_parts[0]
  if (
    repo_name in {"", ".", ".."}
    or owner_name in {".", ".."}
    or not GITHUB_NAME_PATTERN.fullmatch(repo_name)
    or not GITHUB_NAME_PATTERN.fullmatch(owner_name)
  ):
    return {
      "success": False,
      "error": "GitHub owner or repository name is invalid",
    }

  target = WORKSPACE / repo_name

  if target.exists():
    if not target.is_dir() or not (target / ".git").is_dir():
      return {
        "success": False,
        "error": f"Clone target exists but is not a Git repository: {target}",
      }

    return {
        "success": True,
        "repo_path": str(target.resolve()),
        "message": "Repository already exists"
    }
  
  command = [
    "git",
    "clone",
    url,
    str(target.resolve())
  ]

  try:
    result = subprocess.run(
      command,
      capture_output=True,
      text=True,
      timeout=300,
    )
  except subprocess.TimeoutExpired:
    return {
      "success": False,
      "error": "Repository clone timed out after 300 seconds",
    }
  except OSError as error:
    return {
      "success": False,
      "error": f"Unable to run git: {error}",
    }

  if result.returncode != 0:
    return {
    "success": False,
    "error": result.stderr.strip() or "Git clone failed"
    }
  
  return {
    "success": True,

    "repo_path":
        str(target.resolve()),

    "message":
        "Repository cloned successfully"
  }
