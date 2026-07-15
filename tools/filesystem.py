import os
from pathlib import Path
from collections import Counter

MAX_LIST_FILES = 2000
MAX_READ_CHARS = 10000
MAX_SEARCH_RESULTS = 100
MIN_README_CHARS = 100
MAX_README_CHARS = 10000

IGNORED_DIRECTORIES = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
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


def is_ignored(path: Path) -> bool:
    return any(part in IGNORED_DIRECTORIES for part in path.parts)


def iter_repository_files(root: Path):
    for current_dir, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(current_dir)
        relative_dir = current.relative_to(root)

        dirnames[:] = [
            name
            for name in dirnames
            if not is_ignored(relative_dir / name) and not (current / name).is_symlink()
        ]

        for filename in filenames:
            file = current / filename
            relative_path = file.relative_to(root)

            if is_ignored(relative_path):
                continue

            try:
                file.resolve().relative_to(root)
            except (OSError, ValueError):
                continue

            if file.is_file():
                yield file, relative_path


def list_files(repo_path: str, max_files: int = 200):
    root = Path(repo_path).resolve()

    if not root.exists():
        return {"success": False, "error": f"Path does not exist: {repo_path}"}

    if not root.is_dir():
        return {
            "success": False,
            "error": f"Path is not a directory: {repo_path}",
        }

    if not 1 <= max_files <= MAX_LIST_FILES:
        return {
            "success": False,
            "error": f"max_files must be between 1 and {MAX_LIST_FILES}",
        }

    files = []

    for _, relative_path in iter_repository_files(root):
        files.append(relative_path.as_posix())

    files = sorted(files)
    total_files = len(files)
    truncated = False

    if len(files) > max_files:
        files = files[:max_files]
        truncated = True

    return {
        "success": True,
        "repo_path": str(root),
        "files": files,
        "total_files": total_files,
        "returned_files": len(files),
        "truncated": truncated,
    }


def read_file(repo_path: str, file_path: str, max_chars: int = 8000):
    root = Path(repo_path).resolve()

    if not root.exists():
        return {
            "success": False,
            "error": (f"Repository path not found: " f"{repo_path}"),
        }

    if not root.is_dir():
        return {
            "success": False,
            "error": (f"Repository path is not a directory: " f"{repo_path}"),
        }

    if not 1 <= max_chars <= MAX_READ_CHARS:
        return {
            "success": False,
            "error": f"max_chars must be between 1 and {MAX_READ_CHARS}",
        }

    path = (root / file_path).resolve()
    try:
        path.relative_to(root)

    except ValueError:
        return {"success": False, "error": "File path must stay inside the repository"}

    if not path.exists():
        return {"success": False, "error": f"File not found: {file_path}"}

    if not path.is_file():
        return {"success": False, "error": f"Path is not a file: {file_path}"}

    try:
        content = path.read_text(encoding="utf-8")

        truncated = False

        if len(content) > max_chars:
            content = content[:max_chars]
            truncated = True

    except UnicodeDecodeError:
        return {
            "success": False,
            "error": (f"File is not valid UTF-8 text: " f"{file_path}"),
        }
    except PermissionError:
        return {
            "success": False,
            "error": (f"Permission denied: {file_path}"),
        }
    except OSError as error:
        return {
            "success": False,
            "error": str(error),
        }

    return {
        "success": True,
        "repo_path": str(root),
        "file_path": path.relative_to(root).as_posix(),
        "content": content,
        "truncated": truncated,
    }


def search_code(repo_path: str, keyword: str, max_results: int = 20):
    root = Path(repo_path).resolve()

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

    if not 1 <= max_results <= MAX_SEARCH_RESULTS:
        return {
            "success": False,
            "error": (f"max_results must be between 1 and {MAX_SEARCH_RESULTS}"),
        }

    matches = []
    normalized_keyword = keyword.lower()

    for file, relative_path in iter_repository_files(root):
        try:
            content = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError, OSError):
            continue

        lines = content.splitlines()

        for line_number, line in enumerate(lines, start=1):
            if normalized_keyword not in line.lower():
                continue

            matches.append(
                {
                    "file_path": relative_path.as_posix(),
                    "line": line_number,
                    "content": line.strip(),
                }
            )

            if len(matches) >= max_results:
                return {
                    "success": True,
                    "repo_path": str(root),
                    "keyword": keyword,
                    "matches": matches,
                    "truncated": True,
                }

    return {
        "success": True,
        "repo_path": str(root),
        "keyword": keyword,
        "matches": matches,
        "truncated": False,
    }


def summarize_repository(repo_path: str, readme_max_chars: int = 2000):
    root = Path(repo_path).resolve()

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

    if not MIN_README_CHARS <= readme_max_chars <= MAX_README_CHARS:
        return {
            "success": False,
            "error": (
                "readme_max_chars must be between "
                f"{MIN_README_CHARS} and {MAX_README_CHARS}"
            ),
        }

    files = []
    language_counts = Counter()
    extension_counts = Counter()
    important_files = []

    for file, relative_path in iter_repository_files(root):
        files.append(relative_path)

        suffix = file.suffix.lower()

        if suffix:
            extension_counts[suffix] += 1

        language = LANGUAGE_BY_EXTENSION.get(suffix)

        if language:
            language_counts[language] += 1

        if file.name.lower() in IMPORTANT_FILENAMES:
            important_files.append(relative_path.as_posix())

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
        "repo_name": root.name,
        "repo_path": str(root),
        "total_files": len(files),
        "top_level_structure": top_level_structure,
        "languages": dict(language_counts.most_common()),
        "extensions": dict(extension_counts.most_common()),
        "important_files": sorted(important_files),
        "readme_path": (
            readme_path.relative_to(root).as_posix()
            if readme_path is not None
            else None
        ),
        "readme_preview": readme_preview,
    }
