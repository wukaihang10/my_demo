import json

from workflows.repository_analysis import (
    analyze_repository,
)


def main():
    repository_url = input(
        "Enter a GitHub repository URL: "
    ).strip()

    if not repository_url:
        print(
            "Repository URL cannot be empty."
        )
        return

    result = analyze_repository(
        repository_url
    )

    print("\n===== Final Answer =====\n")
    print(result["answer"])


    print("\n===== Repository State =====\n")

    print(json.dumps(
        result["state"],
        ensure_ascii = False,
        indent = 2
    ))


    print("\n===== Agent Trace =====\n")
    print(
        json.dumps(
            result["trace"],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()