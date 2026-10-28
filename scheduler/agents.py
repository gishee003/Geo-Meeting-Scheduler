"""Negotiating agents, one per participant.

Each agent knows only its own person's calendar, rules and preferences. Two
implementations share one interface:

  LLMAgent       - talks through an LLM: Gemini or a local Ollama model
                   (the neural half of the system)
  HeuristicAgent - deterministic, offline stand-in used for tests and for
                   running without an API key; it checks proposals against
                   its own person's rules with a one-person solver

Both receive the public Board and return a Proposal or a Reply; the
negotiation loop decides what the referee (full solver) gets to check.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

from .llm import LLM
from .models import Person, Scenario
from .protocol import Board, Proposal, Reply
from .solver import MeetingSolver

SYSTEM_KINDS = {"horizon", "slot_grid", "venue_domain"}
DESCRIBED_ELSEWHERE = {"working_hours", "busy", "reach_venue", "onward_travel"}

# Privacy-preserving reasons an agent gives when it declines a slot.
REASONS = {
    "busy": "already has a commitment then",
    "working_hours": "is outside working hours then",
    "reach_venue": "cannot travel to that venue in time",
    "onward_travel": "would not make the next commitment after the meeting",
    "horizon": "cannot do a slot outside the planning window",
    "slot_grid": "needs a start on the slot grid",
}


def personal_solver(sc: Scenario, pid: str) -> MeetingSolver:
    """A solver that knows only one person's constraints."""
    return MeetingSolver(replace(sc, meeting=replace(sc.meeting, participants=[pid])))


class Agent(ABC):
    def __init__(self, person: Person, scenario: Scenario):
        self.person = person
        self.sc = scenario
        self.me = personal_solver(scenario, person.id)

    @property
    def name(self) -> str:
        return self.person.name

    @abstractmethod
    def propose(self, board: Board) -> Proposal: ...

    @abstractmethod
    def respond(self, board: Board, proposal: Proposal) -> Reply: ...


# ---------------------------------------------------------------------- offline

class HeuristicAgent(Agent):
    """Rule-based negotiator: proposes its own best slots, accepts anything its person can do."""

    def __init__(self, person: Person, scenario: Scenario, options: int = 12):
        super().__init__(person, scenario)
        self.options = [(o.start, o.venue) for o in self.me.suggest(options, min_gap=scenario.slot_minutes)]

    def _ok(self, start: int, venue: str | None) -> bool:
        return self.me.verify(start, venue).valid

    def _next_idea(self, board: Board) -> tuple[int, str | None] | None:
        hints = [s for s in board.suggestions if self._ok(*s)]
        for slot in hints + self.options:
            if slot not in board.tried:
                return slot
        return None

    def propose(self, board: Board) -> Proposal:
        idea = self._next_idea(board) or (self.options[0] if self.options else None)
        if idea is None:
            return Proposal(self.person.id, "", None, f"{self.name} has no free slot at all.")
        start, venue = idea
        return Proposal(self.person.id, board.iso(start), venue,
                        f"How about {board.slot(start, venue)}?", start)

    def respond(self, board: Board, proposal: Proposal) -> Reply:
        v = self.me.verify(proposal.start, proposal.venue)
        if v.valid:
            return Reply(self.person.id, True, f"That works for {self.name}.")
        reason = REASONS.get(v.violations[0].kind, "has a rule that rules this out")
        idea = self._next_idea(board)
        counter = None
        text = f"Sorry, {self.name} {reason}."
        if idea:
            start, venue = idea
            counter = Proposal(self.person.id, board.iso(start), venue, "", start)
            text += f" Could we do {board.slot(start, venue)} instead?"
        return Reply(self.person.id, False, text, counter)


# -------------------------------------------------------------------------- LLM

class LLMAgent(Agent):
    def __init__(self, person: Person, scenario: Scenario, llm: LLM, use_solver: bool = True):
        super().__init__(person, scenario)
        self.llm = llm
        self.use_solver = use_solver
        self.system = self._system_prompt()

    # --- prompts ----------------------------------------------------------------

    def _when(self, minutes: int) -> str:
        sc, tz = self.sc, self.person.tz
        text = sc.fmt(minutes)
        if tz.key != sc.display_tz.key:
            text += f" (your local time {sc.fmt(minutes, tz)})"
        return text

    def _profile(self) -> str:
        sc, p, me = self.sc, self.person, self.me
        wh = p.working_hours
        lines = [
            f"Name: {p.name}",
            f"Home time zone: {p.tz.key}",
            f"Working hours: {', '.join(wh.days)} {wh.start:%H:%M}-{wh.end:%H:%M} local time",
        ]
        if sc.geo_enabled:
            back = f" and must be back there by {p.return_by:%H:%M}" if p.return_by else ""
            lines.append(f"Based at: {sc.location_name(p.base)}. Each day {p.name} can leave no earlier than "
                         f"{p.earliest_departure:%H:%M}{back} (local time).")
        lines.append("Existing calendar (fixed, cannot move):")
        for ev in p.calendar or []:
            where = f" at {sc.location_name(ev.location or p.base)}" if sc.geo_enabled else ""
            lines.append(f"  - {self._when(ev.start)} to {sc.to_datetime(ev.end):%H:%M}: {ev.title}{where}")
        if not p.calendar:
            lines.append("  (nothing booked)")
        rules = [c for c in me.hard if c.kind not in SYSTEM_KINDS | DESCRIBED_ELSEWHERE]
        if rules:
            lines.append("Hard rules (must never be broken):")
            lines += [f"  - {c.text}" for c in rules]
        if me.soft:
            lines.append("Preferences (nice to have; higher importance matters more):")
            lines += [f"  - {c.text} (importance {c.weight})" for c in me.soft]
        if sc.geo_enabled:
            lines.append(f"One-way travel from {sc.location_name(p.base)} to each candidate venue:")
            for v, minutes in zip(sc.meeting.venues, me.cost[p.id]):
                if minutes == 0:
                    lines.append(f"  - {sc.location_name(v)} [{v}]: no travel needed")
                    continue
                mode = "flight needed" if sc.travel.needs_flight(p.base, v) else "by road"
                lines.append(f"  - {sc.location_name(v)} [{v}]: about {minutes} min ({mode})")
        return "\n".join(lines)

    def _meeting(self) -> str:
        sc, m = self.sc, self.sc.meeting
        names = ", ".join(sc.people[pid].name for pid in m.participants)
        first, last = sc.to_datetime(0), sc.to_datetime(sc.horizon_end - 1)
        lines = [
            f"Title: {m.title}",
            f"Duration: {m.duration} minutes",
            f"Participants: {names}",
            f"Window: {first:%A %d %B %Y} to {last:%A %d %B %Y} ({sc.horizon_days} days)",
            f"Times are written in {sc.display_tz.key} as YYYY-MM-DDTHH:MM (for example "
            f"{first:%Y-%m-%d}T10:00); starts must be on a {sc.slot_minutes}-minute boundary.",
        ]
        if sc.geo_enabled:
            venues = ", ".join(f"{sc.location_name(v)} [{v}]" for v in m.venues)
            lines.append(f"Candidate venues (use the id in brackets): {venues}")
        return "\n".join(lines)

    def _system_prompt(self) -> str:
        p = self.person
        if self.use_solver:
            checking = ("A referee (a logic solver) checks every proposal against everyone's calendars, rules "
                        "and travel before it is put to the group. A slot the referee marks INVALID cannot be "
                        "used; read its reasons and do not propose that slot again.")
        else:
            checking = (f"Nobody checks proposals automatically. Before proposing or accepting a slot, make sure "
                        f"it really works for {p.name}, including travel time if venues are involved.")
        return f"""You are the scheduling assistant for {p.name}. You negotiate with the assistants of the other \
participants to agree on a time{' and venue' if self.sc.geo_enabled else ''} for a meeting.

Represent {p.name}: never agree to something that breaks one of {p.name}'s hard rules or clashes with their \
calendar or travel. Preferences are nice to have; be willing to give them up so the group can agree.
{checking}
Keep messages to the group short (one to three sentences) and friendly. Protect privacy: say "I'm busy then" \
rather than naming meetings.

# Meeting
{self._meeting()}

# What you know about {p.name} (private to you)
{self._profile()}"""

    def _venue_schema(self, allow_empty: bool) -> dict:
        venues = list(self.sc.meeting.venues) + ([""] if allow_empty else [])
        return {"type": "string", "enum": venues}

    def propose(self, board: Board) -> Proposal:
        geo = self.sc.geo_enabled
        props = {"message": {"type": "string"}, "start": {"type": "string"}}
        if geo:
            props["venue"] = self._venue_schema(False)
        schema = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
        task = (f"It is your turn to propose a slot. Choose a start time"
                f"{' and a venue id' if geo else ''} that works for {self.name} and, judging from the "
                f"conversation, is likely to work for the others. Do not repeat a slot that was already rejected. "
                f"'message' is your short note to the group.")
        if board.suggestions:
            task += " The referee has suggested some valid slots above; you may pick one of them."
        out = self.llm.json(self.system, f"# Conversation so far\n{board.render()}\n\n# Your task\n{task}", schema)
        venue = out.get("venue") if geo else None
        return Proposal(self.person.id, out["start"], venue, out["message"])

    def respond(self, board: Board, proposal: Proposal) -> Reply:
        geo = self.sc.geo_enabled
        props = {
            "decision": {"type": "string", "enum": ["accept", "reject"]},
            "message": {"type": "string"},
            "counter_start": {"type": "string"},
        }
        if geo:
            props["counter_venue"] = self._venue_schema(True)
        schema = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
        who = self.sc.people[proposal.by].name if proposal.by in self.sc.people else "The referee"
        status = (" The referee has confirmed it breaks nobody's hard constraints." if self.use_solver else "")
        task = (f"{who} proposed {board.slot(proposal.start, proposal.venue)}.{status} Decide for {self.name}: "
                f"accept if it fits all of {self.name}'s hard rules, calendar and travel (giving up preferences is "
                f"fine). If you reject, you may suggest a different slot in counter_start"
                f"{'/counter_venue' if geo else ''}; otherwise leave {'them' if geo else 'it'} empty. "
                f"'message' is your short reply to the group.")
        out = self.llm.json(self.system, f"# Conversation so far\n{board.render()}\n\n# Your task\n{task}", schema)
        accept = out["decision"] == "accept"
        counter = None
        if not accept and out.get("counter_start", "").strip():
            counter = Proposal(self.person.id, out["counter_start"], out.get("counter_venue") or None if geo else None)
        return Reply(self.person.id, accept, out["message"], counter)
