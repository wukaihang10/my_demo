from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class ToolHistory:
    records: list[ToolCallRecord] = field(default_factory=list)

    def add(self, tool_name: str, arguments: dict[str, Any]):
        self.records.append(ToolCallRecord(tool_name=tool_name, arguments=arguments))

    def count(self, tool_name: str, arguments: dict[str, Any]) -> int:
        return sum(
            1
            for record in self.records
            if (record.tool_name == tool_name and record.arguments == arguments)
        )

    def repeated(
        self, tool_name: str, arguments: dict[str, Any], threshold: int = 2
    ) -> bool:
        return self.count(tool_name, arguments) >= threshold
