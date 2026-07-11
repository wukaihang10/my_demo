from pathlib import Path
from collections import Counter

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
  
IGNORED_DIRECTORIES = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
}

def search_code(repo_path, keyword, max_results=20):
    root = Path(repo_path)

    if not root.exists():
        return {
            "success": False,
            "error": f"Path not found: {repo_path}",
        }

    if not root.is_dir():
        return {
            "success": False,
            "error": f"Path is not a directory: {repo_path}",
        }

    if not keyword.strip():
        return {
            "success": False,
            "error": "Keyword cannot be empty",
        }

    matches = []
    normalized_keyword = keyword.lower()

    for file in root.rglob("*"):
        if not file.is_file():
            continue

        if any(part in IGNORED_DIRECTORIES for part in file.parts):
            continue

        try:
            content = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError, OSError):
            continue

        lines = content.splitlines()

        for line_number, line in enumerate(lines, start=1):
            if normalized_keyword not in line.lower():
                continue

            matches.append({
                "file": str(file),
                "line": line_number,
                "content": line.strip(),
            })

            if len(matches) >= max_results:
                return {
                    "success": True,
                    "keyword": keyword,
                    "matches": matches,
                    "truncated": True,
                }

    return {
        "success": True,
        "keyword": keyword,
        "matches": matches,
        "truncated": False,
    }


LANGUAGE_BY_EXTENSION = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".java": "Java",
    ".c": "C",
    ".h": "C/C++ Header",
    ".cpp": "C++",
    ".cc": "C++",
    ".go": "Go",
    ".rs": "Rust",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".cs": "C#",
    ".sh": "Shell",
    ".sql": "SQL",
    ".html": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".md": "Markdown",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".xml": "XML",
}

IMPORTANT_FILENAMES = {
    "readme.md",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".gitignore",
    "license",
    "license.md",
    "makefile",
    "main.py",
    "app.py",
    "manage.py",
}

def summarize_repository(repo_path, readme_max_chars=2000):
    root = Path(repo_path)

    if not root.exists():
        return {
            "success": False,
            "error": f"Path not found: {repo_path}",
        }

    if not root.is_dir():
        return {
            "success": False,
            "error": f"Path is not a directory: {repo_path}",
        }

    files = []
    language_counts = Counter()
    extension_counts = Counter()
    important_files = []

    for file in root.rglob("*"):
        if not file.is_file():
            continue

        if any(part in IGNORED_DIRECTORIES for part in file.parts):
            continue

        relative_path = file.relative_to(root)
        files.append(relative_path)

        suffix = file.suffix.lower()

        if suffix:
            extension_counts[suffix] += 1

        language = LANGUAGE_BY_EXTENSION.get(suffix)

        if language:
            language_counts[language] += 1

        if file.name.lower() in IMPORTANT_FILENAMES:
            important_files.append(str(relative_path))

    top_level_structure = sorted(
        item.name + ("/" if item.is_dir() else "")
        for item in root.iterdir()
        if item.name not in IGNORED_DIRECTORIES
    )

    readme_preview = None
    readme_path = None

    for candidate_name in ("README.md", "readme.md", "README", "Readme.md"):
        candidate = root / candidate_name

        if candidate.is_file():
            readme_path = candidate
            break

    if readme_path is not None:
        try:
            readme_content = readme_path.read_text(encoding="utf-8")
            readme_preview = readme_content[:readme_max_chars]
        except (UnicodeDecodeError, PermissionError, OSError):
            readme_preview = None

    return {
        "success": True,
        "repository_name": root.name,
        "repository_path": str(root),
        "total_files": len(files),
        "top_level_structure": top_level_structure,
        "languages": dict(language_counts.most_common()),
        "extensions": dict(extension_counts.most_common()),
        "important_files": sorted(important_files),
        "readme_preview": readme_preview,
    }
