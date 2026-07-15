from .base import Tool
from .github import clone_repository
from .filesystem import list_files, read_file, search_code, summarize_repository

TOOLS = [
    Tool(
        name="clone_repository",
        description="Clone a Github repository to local workspace.",
        function=clone_repository,
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Github repository URL"}
            },
            "required": ["url"],
        },
    ),
    Tool(
        name="summarize_repository",
        description="""
Collect a structured overview of a local repository.

Use this after cloning a repository to quickly understand its
top-level structure, file types, languages, important files,
and README introduction.
""",
        function=summarize_repository,
        parameters={
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Local repository path",
                },
                "readme_max_chars": {
                    "type": "integer",
                    "description": ("Maximum number of README characters to return"),
                    "default": 2000,
                    "minimum": 100,
                    "maximum": 10000,
                },
            },
            "required": [
                "repo_path",
            ],
        },
    ),
    Tool(
        name="list_files",
        description="""
List all files in a GitHub repository.
Use this to understand repository structure.
""",
        function=list_files,
        parameters={
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Local repository path"},
                "max_files": {
                    "type": "integer",
                    "description": "Maximum number of file paths to return",
                    "default": 200,
                    "minimum": 1,
                    "maximum": 2000,
                },
            },
            "required": ["repo_path"],
        },
    ),
    Tool(
        name="read_file",
        description="""
Read a UTF-8 text file inside a local repository.

The file_path must be relative to repo_path.
Use paths returned by list_files, search_code,
or summarize_repository.
""",
        function=read_file,
        parameters={
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Local path to the repository root",
                },
                "file_path": {
                    "type": "string",
                    "description": "File path relative to the " "repository root",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum number of file characters to return",
                    "default": 8000,
                    "minimum": 1,
                    "maximum": 10000,
                },
            },
            "required": [
                "repo_path",
                "file_path",
            ],
        },
    ),
    Tool(
        name="search_code",
        description="""
Search for a keyword in repository files.

Use this when you need to find
where a feature is implemented.
""",
        function=search_code,
        parameters={
            "type": "object",
            "properties": {
                "repo_path": {"type": "string", "description": "Repository path"},
                "keyword": {"type": "string", "description": "Keyword to search"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matching lines to return",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 100,
                },
            },
            "required": ["repo_path", "keyword"],
        },
    ),
]

TOOL_MAP = {tool.name: tool for tool in TOOLS}
