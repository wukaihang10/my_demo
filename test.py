from pprint import pprint

from tools.filesystem import summarize_repository


result = summarize_repository(
    repo_path="workspace/repos/my_demo",
)

pprint(result)