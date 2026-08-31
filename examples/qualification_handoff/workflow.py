from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QualificationState:
    """Application-owned state behind the recipe's two client tools."""

    required_fields: tuple[str, ...] = ("need", "region", "timeframe")
    answers: dict[str, str] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    handoff_started: bool = False
    handoff_reference: str = "demo-handoff-001"

    def record(self, args: dict[str, Any]) -> dict[str, Any]:
        key = args.get("field")
        value = args.get("value")
        if key not in self.required_fields:
            raise ValueError("field must name a configured qualification field")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("value must be non-empty text")

        self.answers[str(key)] = value.strip()
        missing = [item for item in self.required_fields if item not in self.answers]
        self.events.append({
            "type": "qualification_recorded",
            "field": key,
            "complete": not missing,
        })
        return {
            "recorded": key,
            "missing_required": missing,
            "complete": not missing,
        }

    def start_handoff(self, args: dict[str, Any]) -> dict[str, Any]:
        missing = [item for item in self.required_fields if item not in self.answers]
        if missing:
            raise ValueError(
                f"qualification is incomplete; missing: {', '.join(missing)}"
            )
        summary = args.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("summary must be non-empty text")
        if self.handoff_started:
            return {
                "handoff_requested": True,
                "handoff_reference": self.handoff_reference,
                "duplicate": True,
            }

        self.handoff_started = True
        self.events.append({
            "type": "handoff_requested",
            "summary": summary.strip(),
        })
        return {
            "handoff_requested": True,
            "handoff_reference": self.handoff_reference,
        }
