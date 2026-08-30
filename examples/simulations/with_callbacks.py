from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from converse_recipes import SimulationCase, run_simulation


async def main() -> None:
    load_dotenv()
    source = Path(__file__).with_name("appointment_booking.json")
    case = SimulationCase.from_dict(json.loads(source.read_text()))

    async def lookup(args):
        assert args.get("day")
        return {"slots": [{"slot_id": "tue-1530", "time": "3:30pm", "price": "€85"}]}

    def book(args):
        # The case checks that this code reaches the caller, so answer with the same one.
        return {"booked": True, "confirmation": "PT-2048", "slot_id": args["slot_id"]}

    case = replace(case, fixtures={"check_availability": lookup, "book_appointment": book})
    report = await run_simulation(
        os.environ.get("CONVERSE_URL", "wss://converse.trelis.com/ws"),
        os.environ["CONVERSE_API_KEY"], case, modality="text",
    )
    print(json.dumps({
        "passed": report.passed,
        "checks": report.check_results,
        "transcript": report.transcript,
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
