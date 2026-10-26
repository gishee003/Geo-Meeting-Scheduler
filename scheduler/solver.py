"""Z3 encoding of the meeting-scheduling problem.

Decision variable
    S : Int  -- meeting start, in minutes since the horizon start (UTC)

The meeting occupies [S, S + D) where D is the meeting duration.

Every rule becomes a *labelled* Z3 formula over S. Labels let us
  * verify a proposal and list exactly which rules it breaks,
  * explain those failures in plain language (for the LLM agents later),
  * extract an unsat core when no slot exists at all.

Hard rules must hold. Soft preferences are weighted and maximised with
z3.Optimize (weighted MaxSMT).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time

import z3

from .models import DAY_NAMES, Person, Rule, Scenario


@dataclass
class Constraint:
    id: str
    kind: str          # e.g. "busy", "working_hours", "prefer_time_window"
    person: str | None  # person id, or None for system rules
    text: str          # human-readable description
    expr: z3.BoolRef
    weight: int = 0    # > 0 only for soft preferences


@dataclass
class Verdict:
    start: int
    valid: bool
    violations: list[Constraint]
    prefs_met: list[Constraint]
    prefs_missed: list[Constraint]

    @property
    def score(self) -> int:
        return sum(c.weight for c in self.prefs_met)


@dataclass
class Option:
    start: int
    verdict: Verdict


@dataclass
class Diagnosis:
    feasible: bool
    conflict: list[Constraint] = field(default_factory=list)


def _overlaps(S, D, a: int, b: int) -> z3.BoolRef:
    """Meeting [S, S+D) overlaps interval [a, b)."""
    return z3.And(S < b, S + D > a)


def _inside(S, D, a: int, b: int) -> z3.BoolRef:
    """Meeting [S, S+D) lies fully inside [a, b)."""
    return z3.And(S >= a, S + D <= b)


def _hhmm(t: time) -> str:
    return t.strftime("%H:%M")


class MeetingSolver:
    def __init__(self, scenario: Scenario):
        self.sc = scenario
        self.S = z3.Int("start")
        self.D = scenario.meeting.duration
        self.hard: list[Constraint] = []
        self.soft: list[Constraint] = []
        self._build()

    # ------------------------------------------------------------------ build

    def _build(self) -> None:
        S, D, sc = self.S, self.D, self.sc

        self.hard.append(Constraint(
            "horizon", "horizon", None,
            f"The meeting must fall within the planning horizon ({sc.horizon_days} days from "
            f"{sc.fmt(0)})",
            z3.And(S >= 0, S + D <= sc.horizon_end),
        ))
        self.hard.append(Constraint(
            "slot_grid", "slot_grid", None,
            f"The meeting must start on a {sc.slot_minutes}-minute boundary",
            S % sc.slot_minutes == 0,
        ))

        for person in sc.participants():
            self._build_person(person)

    def _windows(self, person: Person, start: time, end: time, days: list[str] | None = None):
        """Horizon-minute intervals of a daily local clock window for a person."""
        out = []
        for d, _ in self.sc.local_days(person.tz):
            if days is None or DAY_NAMES[d.weekday()] in days:
                out.append(self.sc.local_window(d, start, end, person.tz))
        return out

    def _day_spans(self, person: Person, days: list[str]):
        """Horizon-minute intervals covering whole local days with the given names."""
        return [(m, m + 24 * 60) for d, m in self.sc.local_days(person.tz) if DAY_NAMES[d.weekday()] in days]

    def _build_person(self, p: Person) -> None:
        S, D, sc = self.S, self.D, self.sc
        wh = p.working_hours

        # Working hours: the meeting sits inside one working window.
        windows = self._windows(p, wh.start, wh.end, wh.days)
        self.hard.append(Constraint(
            f"{p.id}:working_hours", "working_hours", p.id,
            f"{p.name} works {', '.join(wh.days)} {_hhmm(wh.start)}-{_hhmm(wh.end)} ({p.tz.key})",
            z3.Or([_inside(S, D, a, b) for a, b in windows]) if windows else z3.BoolVal(False),
        ))

        # Existing calendar events: no overlap.
        for i, ev in enumerate(p.calendar):
            self.hard.append(Constraint(
                f"{p.id}:busy:{i}", "busy", p.id,
                f"{p.name} is busy {sc.fmt(ev.start, p.tz)}-{sc.to_datetime(ev.end, p.tz):%H:%M} ({ev.title})",
                z3.Not(_overlaps(S, D, ev.start, ev.end)),
            ))

        for i, rule in enumerate(p.hard_constraints):
            self.hard.append(self._encode_rule(p, rule, f"{p.id}:hard:{i}"))

        for i, rule in enumerate(p.preferences):
            self.soft.append(self._encode_rule(p, rule, f"{p.id}:pref:{i}"))

    def _encode_rule(self, p: Person, rule: Rule, cid: str) -> Constraint:
        """Encode a hard constraint or soft preference. Same rule types work for both."""
        S, D, sc = self.S, self.D, self.sc
        t, prm = rule.type, rule.params

        if t in ("only_between", "prefer_time_window"):
            start, end = time.fromisoformat(prm["start"]), time.fromisoformat(prm["end"])
            windows = self._windows(p, start, end)
            expr = z3.Or([_inside(S, D, a, b) for a, b in windows])
            verb = "is only available" if t == "only_between" else "prefers meetings"
            text = f"{p.name} {verb} between {_hhmm(start)} and {_hhmm(end)} ({p.tz.key})"

        elif t in ("blocked_window", "avoid_window"):
            start, end = time.fromisoformat(prm["start"]), time.fromisoformat(prm["end"])
            days = prm.get("days")
            windows = self._windows(p, start, end, days)
            expr = z3.And([z3.Not(_overlaps(S, D, a, b)) for a, b in windows])
            when = f" on {', '.join(days)}" if days else " every day"
            verb = "cannot meet" if t == "blocked_window" else "prefers not to meet"
            label = f" ({prm['label']})" if "label" in prm else ""
            text = f"{p.name} {verb} between {_hhmm(start)} and {_hhmm(end)}{when}{label}"

        elif t in ("no_meetings_on_days", "avoid_days"):
            days = prm["days"]
            expr = z3.And([z3.Not(_overlaps(S, D, a, b)) for a, b in self._day_spans(p, days)])
            verb = "has no meetings" if t == "no_meetings_on_days" else "prefers no meetings"
            text = f"{p.name} {verb} on {', '.join(days)}"

        elif t == "prefer_days":
            days = prm["days"]
            expr = z3.Or([_inside(S, D, a, b) for a, b in self._day_spans(p, days)])
            text = f"{p.name} prefers meetings on {', '.join(days)}"

        elif t == "unavailable":
            a, b = sc.parse_local(prm["start"], p.tz), sc.parse_local(prm["end"], p.tz)
            expr = z3.Not(_overlaps(S, D, a, b))
            reason = f" ({prm['label']})" if "label" in prm else ""
            text = f"{p.name} is unavailable {sc.fmt(a, p.tz)} - {sc.fmt(b, p.tz)}{reason}"

        elif t == "buffer_around_events":
            m = int(prm["minutes"])
            expr = z3.And([z3.Not(_overlaps(S, D, ev.start - m, ev.end + m)) for ev in p.calendar])
            text = f"{p.name} prefers a {m}-minute gap before and after other meetings"

        else:
            raise ValueError(f"Unknown rule type '{t}' for {p.id}")

        return Constraint(cid, t, p.id, text, z3.simplify(expr), weight=rule.weight)

    # ---------------------------------------------------------------- queries

    def _holds(self, c: Constraint, start: int) -> bool:
        """Evaluate a constraint's formula at S = start."""
        return z3.is_true(z3.simplify(z3.substitute(c.expr, (self.S, z3.IntVal(start)))))

    def verify(self, start: int) -> Verdict:
        """Check a proposed start time against every hard rule and preference."""
        violations = [c for c in self.hard if not self._holds(c, start)]
        met = [c for c in self.soft if self._holds(c, start)]
        missed = [c for c in self.soft if c not in met]
        return Verdict(start, not violations, violations, met, missed)

    def suggest(self, k: int = 3, min_gap: int | None = None) -> list[Option]:
        """Best k valid slots, ranked by preference score (then earliest).

        Successive options are at least `min_gap` minutes apart (default: the
        meeting duration) so the list is not 15-minute shifts of one slot.
        """
        gap = self.D if min_gap is None else min_gap
        opt = z3.Optimize()
        for c in self.hard:
            opt.add(c.expr)
        for c in self.soft:
            opt.add_soft(c.expr, c.weight)
        opt.minimize(self.S)  # tie-break: earliest; lower priority than soft goals

        options = []
        for _ in range(k):
            if opt.check() != z3.sat:
                break
            start = opt.model()[self.S].as_long()
            options.append(Option(start, self.verify(start)))
            opt.add(z3.Or(self.S <= start - gap, self.S >= start + gap))
        return options

    def diagnose(self) -> Diagnosis:
        """If no valid slot exists, return a minimal set of clashing hard rules."""
        s = z3.Solver()
        s.set("core.minimize", True)
        tracked = {}
        for c in self.hard:
            lit = z3.Bool(c.id)
            tracked[c.id] = c
            s.assert_and_track(c.expr, lit)
        if s.check() == z3.sat:
            return Diagnosis(True)
        return Diagnosis(False, [tracked[str(lit)] for lit in s.unsat_core()])

    # ----------------------------------------------------------- explanation

    def explain(self, verdict: Verdict) -> str:
        """Plain-language explanation of a verdict (later fed back to agents)."""
        sc = self.sc
        slot = f"{sc.fmt(verdict.start)}-{sc.to_datetime(verdict.start + self.D):%H:%M} ({sc.display_tz.key})"
        lines = []
        if verdict.valid:
            lines.append(f"VALID: {slot} satisfies every hard constraint.")
        else:
            lines.append(f"INVALID: {slot} breaks {len(verdict.violations)} hard constraint(s):")
            lines += [f"  - {c.text}" for c in verdict.violations]
        if self.soft:
            total = sum(c.weight for c in self.soft)
            lines.append(f"Preference score: {verdict.score}/{total}")
            lines += [f"  + {c.text} (w={c.weight})" for c in verdict.prefs_met]
            lines += [f"  - {c.text} (w={c.weight}) [not met]" for c in verdict.prefs_missed]
        return "\n".join(lines)

    def explain_diagnosis(self, diag: Diagnosis) -> str:
        if diag.feasible:
            return "A valid meeting slot exists."
        lines = ["NO VALID SLOT EXISTS. These hard constraints cannot all be satisfied together:"]
        lines += [f"  - {c.text}" for c in diag.conflict]
        return "\n".join(lines)
