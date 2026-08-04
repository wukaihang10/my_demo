from __future__ import annotations

import json
from pathlib import Path


class FrozenQueryExpander:
    """
    Evaluation-only QueryExpander.

    使用预先保存的 query rewrites，
    保证 Retrieval 实验可复现。

    Production 仍然使用 LLMQueryExpander。
    """

    def __init__(
        self,
        expansions: dict[
            str,
            tuple[str, ...],
        ],
    ) -> None:
        self.expansions = expansions

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> "FrozenQueryExpander":
        source_path = Path(path)

        with source_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        if not isinstance(payload, list):
            raise ValueError("Query expansion dataset " "must contain a JSON list")

        expansions: dict[
            str,
            tuple[str, ...],
        ] = {}

        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("Each query expansion " "must be an object")

            query = item.get("query")
            rewrites = item.get("rewrites")

            if not isinstance(query, str) or not query.strip():
                raise ValueError("Query must be a " "non-empty string")

            if not isinstance(
                rewrites,
                list,
            ):
                raise ValueError("rewrites must be a list")

            normalized_rewrites: list[str] = []

            for rewrite in rewrites:
                if not isinstance(
                    rewrite,
                    str,
                ):
                    raise ValueError("Each rewrite must " "be a string")

                rewrite = rewrite.strip()

                if rewrite:
                    normalized_rewrites.append(rewrite)

            expansions[query.strip()] = tuple(normalized_rewrites)

        return cls(expansions)

    def expand(
        self,
        query: str,
    ) -> list[str]:
        normalized_query = query.strip()

        return list(
            self.expansions.get(
                normalized_query,
                (),
            )
        )
