from pathlib import Path

from agent.agent import Agent
from rag.repository_manager import (
    RepositoryKnowledgeManager,
)
from tasks.repository import (
    create_repository_task,
)


def main() -> None:
    repository_path = Path(".").resolve()

    repository_manager = RepositoryKnowledgeManager(
        retrieval_mode="quality",
        show_progress_bar=False,
    )

    task = create_repository_task(repository_manager=(repository_manager))

    agent = Agent(task=task)

    user_input = f"""
请分析这个本地 Python 仓库：

{repository_path}

请重点回答：

1. 这个 Agent 的一次完整运行流程是什么？
2. RepositoryKnowledgeManager、PythonRepositoryRAG
   和 RAGIndexStorage 分别负责什么？
3. 仓库知识检索从用户自然语言问题开始，
   到最终 evidence 返回给 Agent，
   中间经过哪些核心组件？

请基于实际代码回答。
不要只根据类名猜测。
必要时使用 repository knowledge search，
并在检索片段不足时读取相关源码确认。
""".strip()

    answer = agent.run(
        user_input,
        max_steps=15,
        max_tool_calls=50,
    )

    print()
    print("===== FINAL ANSWER =====")
    print(answer)

    print()
    print("===== OUTCOME =====")
    print(agent.last_outcome)


if __name__ == "__main__":
    main()
