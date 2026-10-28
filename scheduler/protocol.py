"""Shared negotiation board: proposals, replies and referee messages.

Everything posted here is *public* — every agent sees it. Each agent's
calendar and preferences stay private inside that agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .models import Scenario


@dataclass
class Proposal:
    by: str                  # person id (or "referee")
    start_text: str          # "YYYY-MM-DDTHH:MM" in the scenario time zone, as written by the agent
    venue: str | None
    message: str = ""
    start: int | None = None  # parsed horizon minutes; None if unparseable


@dataclass
class Reply:
    by: str
    accept: bool
    message: str
    counter: Proposal | None = None


@dataclass
class Entry:
    speaker: str
    kind: str   # "proposal", "reply", "referee", "system"
    text: str


@dataclass
class Board:
    sc: Scenario
    entries: list[Entry] = field(default_factory=list)
    suggestions: list[tuple[int, str | None]] = field(default_factory=list)  # referee hints
    tried: set[tuple[int, str | None]] = field(default_factory=set)          # slots already proposed

    def post(self, speaker: str, kind: str, text: str) -> Entry:
        entry = Entry(speaker, kind, text)
        self.entries.append(entry)
        return entry

    def render(self) -> str:
        if not self.entries:
            return "(no messages yet)"
        return "\n".join(f"[{i}] {e.speaker}: {e.text}" for i, e in enumerate(self.entries, 1))

    # --- time helpers -----------------------------------------------------------

    def iso(self, minutes: int) -> str:
        return self.sc.to_datetime(minutes).strftime("%Y-%m-%dT%H:%M")

    def parse(self, text: str) -> int | None:
        """Parse an agent's start time; None if malformed or outside the horizon."""
        try:
            dt = datetime.fromisoformat(text.strip())
        except ValueError:
            return None
        minutes = self.sc.to_minutes(dt.replace(tzinfo=self.sc.display_tz))
        return minutes if 0 <= minutes < self.sc.horizon_end else None

    def slot(self, start: int, venue: str | None) -> str:
        """Human- and machine-readable slot description, e.g. '2026-11-03T14:00 (Tue) at Mumbai office [bom_office]'."""
        when = f"{self.iso(start)} ({self.sc.to_datetime(start):%a})"
        if venue is None:
            return when
        return f"{when} at {self.sc.location_name(venue)} [{venue}]"
