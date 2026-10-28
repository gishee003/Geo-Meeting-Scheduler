"""The negotiation loop: agents propose and reply, the referee (solver) verifies.

Protocol (one round = one proposal evaluated):
  1. If there is no pending counter-proposal, the next agent in turn proposes.
  2. The proposal is parsed. With the solver on, the referee verifies it; an
     INVALID proposal is rejected with the solver's plain-language reasons and
     never reaches the group.
  3. Every other agent accepts or rejects (optionally with a counter-proposal).
     All accept -> agreement. Otherwise the first counter-proposal is evaluated
     next round, or the turn passes to the next agent.
  4. With the solver on, after `suggest_after` rounds without agreement the
     referee posts a few valid slots to help the agents converge; if the
     round limit is reached it falls back to the solver's best slot.

With use_solver=False (the "LLM only" baseline) steps 2 and 4 are skipped,
but the agreed slot is still verified afterwards so results can be scored.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .agents import Agent
from .llm import LLMError
from .models import Scenario
from .protocol import Board, Entry, Proposal
from .solver import MeetingSolver, Verdict


@dataclass
class Outcome:
    status: str                    # "agreed", "impossible", "fallback", "no_agreement", "error"
    start: int | None = None
    venue: str | None = None
    verdict: Verdict | None = None  # solver's check of the final slot (always computed)
    rounds: int = 0
    proposals: int = 0
    invalid_caught: int = 0         # proposals the referee rejected
    error: str | None = None
    transcript: list[Entry] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.verdict is not None and self.verdict.valid


class Negotiation:
    def __init__(self, scenario: Scenario, agents: list[Agent], use_solver: bool = True,
                 max_rounds: int = 8, suggest_after: int = 4,
                 on_post: Callable[[Entry], None] | None = None):
        self.sc = scenario
        self.agents = agents
        self.by_id = {a.person.id: a for a in agents}
        self.use_solver = use_solver
        self.max_rounds = max_rounds
        self.suggest_after = suggest_after
        self.solver = MeetingSolver(scenario)
        self.board = Board(scenario)
        self.on_post = on_post

    def _post(self, speaker: str, kind: str, text: str) -> None:
        entry = self.board.post(speaker, kind, text)
        if self.on_post:
            self.on_post(entry)

    def _finish(self, out: Outcome, status: str, start: int | None = None, venue: str | None = None) -> Outcome:
        out.status, out.start, out.venue = status, start, venue
        if start is not None:
            out.verdict = self.solver.verify(start, venue)
        out.transcript = self.board.entries
        return out

    def _check_format(self, p: Proposal) -> str | None:
        """Problems a proposal has before any reasoning: bad time format or unknown venue."""
        p.start = self.board.parse(p.start_text)
        if p.start is None:
            return (f"could not read the start time '{p.start_text}' (use YYYY-MM-DDTHH:MM inside the planning "
                    f"window)")
        if self.sc.geo_enabled and p.venue not in self.sc.meeting.venues:
            return f"'{p.venue}' is not one of the candidate venues"
        return None

    def run(self) -> Outcome:
        out = Outcome("no_agreement")
        sc = self.sc
        names = ", ".join(a.name for a in self.agents)
        self._post("Mediator", "system",
                   f"Let's schedule '{sc.meeting.title}' ({sc.meeting.duration} min) for {names}.")

        if self.use_solver:
            diag = self.solver.diagnose()
            if not diag.feasible:
                self._post("Referee", "referee", self.solver.explain_diagnosis(diag))
                return self._finish(out, "impossible")

        pending: Proposal | None = None
        turn = 0
        try:
            for rnd in range(1, self.max_rounds + 1):
                out.rounds = rnd
                if pending is None:
                    agent = self.agents[turn % len(self.agents)]
                    turn += 1
                    pending = agent.propose(self.board)
                    self._post(agent.name, "proposal", pending.message or f"I propose {pending.start_text}.")
                proposal, pending = pending, None
                out.proposals += 1

                problem = self._check_format(proposal)
                if problem:
                    self._post("Mediator", "system", f"Proposal not considered: {problem}.")
                    continue
                slot = (proposal.start, proposal.venue)
                self.board.tried.add(slot)

                if self.use_solver:
                    verdict = self.solver.verify(*slot)
                    if not verdict.valid:
                        out.invalid_caught += 1
                        self._post("Referee", "referee", self.solver.explain(verdict, public=True))
                        self._maybe_suggest(rnd)
                        continue
                    self._post("Referee", "referee", self.solver.explain(verdict, public=True))

                replies = []
                for agent in self.agents:
                    if agent.person.id == proposal.by:
                        continue
                    reply = agent.respond(self.board, proposal)
                    replies.append(reply)
                    self._post(agent.name, "reply", reply.message)

                if all(r.accept for r in replies):
                    self._post("Mediator", "system", f"Agreed: {self.board.slot(*slot)}.")
                    return self._finish(out, "agreed", *slot)

                pending = next((r.counter for r in replies if r.counter is not None), None)
                self._maybe_suggest(rnd)
        except LLMError as e:
            out.error = str(e)
            self._post("Mediator", "system", f"Negotiation stopped: {e}")
            return self._finish(out, "error")

        if self.use_solver:
            best = self.solver.suggest(1)[0]
            self._post("Referee", "referee",
                       f"No agreement after {self.max_rounds} rounds. Falling back to the best valid slot: "
                       f"{self.board.slot(best.start, best.venue)}.")
            return self._finish(out, "fallback", best.start, best.venue)
        self._post("Mediator", "system", f"No agreement after {self.max_rounds} rounds.")
        return self._finish(out, "no_agreement")

    def _maybe_suggest(self, rnd: int) -> None:
        if not self.use_solver or self.board.suggestions or rnd < self.suggest_after:
            return
        options = self.solver.suggest(3)
        self.board.suggestions = [(o.start, o.venue) for o in options]
        listing = "; ".join(self.board.slot(o.start, o.venue) for o in options)
        self._post("Referee", "referee", f"Hint - these slots are valid for everyone: {listing}")
