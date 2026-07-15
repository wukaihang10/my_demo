from pprint import pprint
from pathlib import Path

from tools.filesystem import summarize_repository

result = summarize_repository(
    repo_path=Path(__file__).resolve().parent / "workspace" / "repos" / "my_demo",
)

pprint(result)
