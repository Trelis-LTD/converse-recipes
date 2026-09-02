from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator

from dialt import DialtMode, DialtSession, SessionEvent

from .conversation_plan import ConversationPlan


@dataclass
class GuidedAssistant:
    """A thin application controller around one ordinary Dialt session."""

    session: DialtSession
    plan: ConversationPlan
    modality: str
    answers: dict[str, str] = field(default_factory=dict)

    @classmethod
    async def connect(cls, url: str, api_key: str, plan: ConversationPlan, *,
                      modality: str = "text", session_id: str | None = None,
                      greeting: str | bool | None = None) -> "GuidedAssistant":
        session = await DialtSession.connect(
            url, api_key=api_key, session_id=session_id,
            mode=DialtMode(
                modality=modality, instructions=plan.instructions(), tools=[plan.tool()],
                greeting=greeting,
            ),
        )
        return cls(session=session, plan=plan, modality=modality)

    async def send_text(self, text: str) -> None:
        if self.modality != "text":
            raise ValueError("send_text belongs to the text recipe; stream audio in voice mode")
        await self.session.send_text(text)

    async def events(self) -> AsyncIterator[SessionEvent]:
        async for event in self.session.events():
            if event.type == "tool_call" and event.data.get("name") == self.plan.tool_name:
                try:
                    result = self.plan.record(self.answers, event.data.get("args") or {})
                except ValueError as exc:
                    await self.session.send_tool_result(
                        event.data["id"], {"error": str(exc)}, outcome="failed", verified=False)
                else:
                    await self.session.send_tool_result(
                        event.data["id"], result, outcome="succeeded", verified=True)
            yield event

    @property
    def complete(self) -> bool:
        return all(not item.required or item.key in self.answers for item in self.plan.fields)

    async def close(self) -> None:
        await self.session.close()

    async def __aenter__(self) -> "GuidedAssistant":
        return self

    async def __aexit__(self, *_exc) -> None:
        await self.close()
