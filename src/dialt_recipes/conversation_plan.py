from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlanField:
    key: str
    description: str
    required: bool = True

    def __post_init__(self) -> None:
        if not self.key or not self.key.replace("_", "").isalnum():
            raise ValueError("field keys must contain letters, numbers, or underscores")
        if not self.description.strip():
            raise ValueError("field descriptions cannot be empty")


@dataclass(frozen=True)
class ConversationPlan:
    """Declarative evidence contract for a guided conversation recipe.

    The plan does not dictate dialogue order or canned wording. It gives the model a goal and a
    normal client tool for committing structured evidence; the host owns the recorded state.
    """

    name: str
    objective: str
    fields: tuple[PlanField, ...]
    completion: str
    tool_name: str = "record_plan_field"

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.objective.strip() or not self.completion.strip():
            raise ValueError("name, objective, and completion are required")
        keys = [field.key for field in self.fields]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("plans need at least one field and field keys must be unique")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConversationPlan":
        return cls(
            name=str(value["name"]),
            objective=str(value["objective"]),
            fields=tuple(PlanField(
                key=str(item["key"]), description=str(item["description"]),
                required=bool(item.get("required", True)),
            ) for item in value["fields"]),
            completion=str(value["completion"]),
            tool_name=str(value.get("tool_name", "record_plan_field")),
        )

    def instructions(self) -> str:
        evidence = "\n".join(
            f"- {field.key} ({'required' if field.required else 'optional'}): {field.description}"
            for field in self.fields
        )
        return (
            f"You are conducting {self.name}.\n"
            f"Objective: {self.objective}\n\n"
            "Collect the evidence below through a natural conversation. Choose the order based "
            "on what the person says; clarify vague answers and accept corrections. Do not read "
            "the field list aloud. Whenever an answer is sufficiently supported, call "
            f"{self.tool_name}. Update a field if later evidence changes it. Do not claim the "
            "interview is complete until every required field has been recorded.\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Once complete: {self.completion}"
        )

    def tool(self) -> dict[str, Any]:
        keys = [field.key for field in self.fields]
        return {
            "name": self.tool_name,
            "description": (
                "Record or correct one supported piece of evidence from the guided conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "enum": keys},
                    "value": {"type": "string", "description": "Concise evidence in the user's terms."},
                },
                "required": ["field", "value"],
            },
            "read_only": False,
            "expected_duration": "instant",
            "status_label": "interview notes",
        }

    def record(self, answers: dict[str, str], args: dict[str, Any]) -> dict[str, Any]:
        key, value = args.get("field"), args.get("value")
        known = {field.key for field in self.fields}
        if key not in known or not isinstance(value, str) or not value.strip():
            raise ValueError("field must be known and value must be non-empty text")
        answers[str(key)] = value.strip()
        missing = [field.key for field in self.fields if field.required and field.key not in answers]
        return {"recorded": key, "missing_required": missing, "complete": not missing}
