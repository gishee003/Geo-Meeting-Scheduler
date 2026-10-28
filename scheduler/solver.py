"""Z3 encoding of the meeting-scheduling problem.

Decision variables
    S : Int  -- meeting start, in minutes since the horizon start (UTC)
    V : Int  -- index into meeting.venues (only when the meeting has venues)

The meeting occupies [S, S + D) where D is the meeting duration.

Every rule becomes a *labelled* Z3 formula over S (and V). Labels let us
  * verify a proposal and list exactly which rules it breaks,
  * explain those failures in plain language (for the LLM agents later),
  * extract an unsat core when no slot exists at all.

Hard rules must hold. Soft preferences are weighted and maximised with
z3.Optimize (weighted MaxSMT). With geography, total (or worst-case) travel is
minimised next, and the earliest slot wins remaining ties.

Travel (Phase 2)
    Each person's day is a sequence of fixed *anchors*: leaving base in the
    morning, their calendar events (each at a location), and returning to base
    in the evening. The meeting must fall in a gap between two anchors. For
    every gap g and venue v the travel model precomputes
        EA[g, v] = earliest arrival at v after the previous anchor
        LD[g, v] = latest departure from v that still reaches the next anchor
    (flights included), and the constraints are
        in_gap(g) and V = v  ->  S >= EA[g, v]        (reach the venue)
        in_gap(g) and V = v  ->  S + D <= LD[g, v]    (make the next commitment)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import time

import z3

from .geo import Route
from .models import DAY_NAMES, Person, Rule, Scenario

NO_ROUTE_COST = 10_000  # travel cost (minutes) used in the objective when no route exists


@dataclass
class Constraint:
    id: str
    kind: str          # e.g. "busy", "working_hours", "reach_venue", "prefer_time_window"
    person: str | None  # person id, or None for system rules
    text: str          # human-readable description
    expr: z3.BoolRef
    weight: int = 0    # > 0 only for soft preferences
    detail: Callable[[int, int | None], str] | None = None  # proposal-specific explanation


@dataclass
class Anchor:
    label: str
    location: str
    start: int
    end: int


@dataclass
class Gap:
    prev: Anchor
    next: Anchor | None  # None: nothing to reach afterwards
    hi: int              # end of the free gap

    @property
    def lo(self) -> int:
        return self.prev.end


@dataclass
class Trip:
    """How one person gets to the venue and on to their next commitment."""
    person: Person
    gap: Gap | None
    there: Route | None
    onward: Route | None
    plan: Route | None = None   # journey to the venue, leaving as late as possible
    after: Route | None = None  # journey on to the next anchor, leaving right after the meeting


@dataclass
class Verdict:
    start: int
    venue: str | None
    valid: bool
    violations: list[Constraint]
    prefs_met: list[Constraint]
    prefs_missed: list[Constraint]
    travel: dict[str, int] = field(default_factory=dict)  # person id -> one-way travel minutes from base

    @property
    def score(self) -> int:
        return sum(c.weight for c in self.prefs_met)


@dataclass
class Option:
    start: int
    venue: str | None
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
        self.geo = scenario.geo_enabled
        self.venues = scenario.meeting.venues
        self.V = z3.Int("venue") if self.geo else None
        self.hard: list[Constraint] = []
        self.soft: list[Constraint] = []
        self.gaps: dict[str, list[Gap]] = {}
        self.cost: dict[str, list[int]] = {}  # person id -> travel minutes from base, per venue
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
        if self.geo:
            names = ", ".join(sc.location_name(v) for v in self.venues)
            self.hard.append(Constraint(
                "venue_domain", "venue_domain", None,
                f"The venue must be one of: {names}",
                z3.And(self.V >= 0, self.V < len(self.venues)),
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
            where = f" at {sc.location_name(ev.location)}" if self.geo and ev.location else ""
            self.hard.append(Constraint(
                f"{p.id}:busy:{i}", "busy", p.id,
                f"{p.name} is busy {sc.fmt(ev.start, p.tz)}-{sc.to_datetime(ev.end, p.tz):%H:%M} "
                f"({ev.title}){where}",
                z3.Not(_overlaps(S, D, ev.start, ev.end)),
            ))

        if self.geo:
            self._build_travel(p)

        for i, rule in enumerate(p.hard_constraints):
            self.hard.append(self._encode_rule(p, rule, f"{p.id}:hard:{i}"))

        for i, rule in enumerate(p.preferences):
            self.soft.append(self._encode_rule(p, rule, f"{p.id}:pref:{i}"))

    # ------------------------------------------------------------ geography

    def _person_gaps(self, p: Person) -> list[Gap]:
        """Free gaps between a person's fixed anchors, day by day (local time)."""
        sc, base = self.sc, p.base
        gaps = []
        for d, midnight in sc.local_days(p.tz):
            day_start = sc.local_window(d, p.earliest_departure, p.earliest_departure, p.tz)[0]
            anchors = [Anchor(f"start of day at {sc.location_name(base)}", base, day_start, day_start)]
            todays = sorted((e for e in p.calendar if midnight <= e.start < midnight + 24 * 60),
                            key=lambda e: e.start)
            anchors += [Anchor(f"'{e.title}' at {sc.location_name(e.location or base)}",
                               e.location or base, e.start, e.end) for e in todays]
            if p.return_by is not None:
                back = sc.local_window(d, p.return_by, p.return_by, p.tz)[0]
                anchors.append(Anchor(f"being back at {sc.location_name(base)} by {_hhmm(p.return_by)}",
                                      base, back, back))
            gaps += [Gap(a, b, b.start) for a, b in zip(anchors, anchors[1:])]
            if p.return_by is None:
                gaps.append(Gap(anchors[-1], None, midnight + 24 * 60))
        return gaps

    def _in_gap(self, g: Gap) -> z3.BoolRef:
        return _inside(self.S, self.D, g.lo, g.hi)

    def _route_there(self, g: Gap, venue: str) -> Route | None:
        return self.sc.travel.earliest_arrival(g.prev.location, venue, g.prev.end)

    def _route_onward(self, g: Gap, venue: str) -> Route | None:
        if g.next is None:
            return Route(g.hi, g.hi)
        return self.sc.travel.latest_departure(venue, g.next.location, g.next.start)

    def _build_travel(self, p: Person) -> None:
        S, D, V = self.S, self.D, self.V
        gaps = self.gaps[p.id] = self._person_gaps(p)

        reach, onward = [], []
        for g in gaps:
            for i, v in enumerate(self.venues):
                there, on = self._route_there(g, v), self._route_onward(g, v)
                at_v = z3.And(self._in_gap(g), V == i)
                reach.append(z3.Implies(at_v, S >= there.arrive) if there else z3.Not(at_v))
                onward.append(z3.Implies(at_v, S + D <= on.depart) if on else z3.Not(at_v))

        back = f" and be back at base by {_hhmm(p.return_by)}" if p.return_by else ""
        self.hard.append(Constraint(
            f"{p.id}:reach_venue", "reach_venue", p.id,
            f"{p.name} must be able to travel to the venue in time from the previous commitment",
            z3.And(reach), detail=lambda s, v, p=p: self._explain_trip(p, s, v, "there"),
        ))
        self.hard.append(Constraint(
            f"{p.id}:onward_travel", "onward_travel", p.id,
            f"{p.name} must be able to leave the venue in time for the next commitment{back}",
            z3.And(onward), detail=lambda s, v, p=p: self._explain_trip(p, s, v, "onward"),
        ))

        travel = self.sc.travel
        self.cost[p.id] = [m if (m := travel.typical_minutes(p.base, v)) is not None else NO_ROUTE_COST
                           for v in self.venues]

    def _venue_in(self, keep) -> z3.BoolRef:
        idx = [i for i, v in enumerate(self.venues) if keep(v)]
        return z3.Or([self.V == i for i in idx]) if idx else z3.BoolVal(False)

    # ---------------------------------------------------------------- rules

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

        elif t in ("in_person_only", "prefer_in_person", "no_flying", "avoid_flying",
                   "max_travel_minutes", "prefer_max_travel_minutes", "allowed_venues", "prefer_venues"):
            expr, text = self._encode_venue_rule(p, t, prm)

        else:
            raise ValueError(f"Unknown rule type '{t}' for {p.id}")

        return Constraint(cid, t, p.id, text, z3.simplify(expr), weight=rule.weight)

    def _encode_venue_rule(self, p: Person, t: str, prm: dict) -> tuple[z3.BoolRef, str]:
        if not self.geo:
            raise ValueError(f"Rule '{t}' for {p.id} needs a meeting with venues")
        travel, sc = self.sc.travel, self.sc

        if t in ("in_person_only", "prefer_in_person"):
            verb = "must attend" if t == "in_person_only" else "prefers to attend"
            return self._venue_in(lambda v: not travel.loc[v].online), f"{p.name} {verb} in person"

        if t in ("no_flying", "avoid_flying"):
            verb = "cannot fly" if t == "no_flying" else "prefers not to fly"
            return (self._venue_in(lambda v: not travel.needs_flight(p.base, v)),
                    f"{p.name} {verb} (travels from {sc.location_name(p.base)})")

        if t in ("max_travel_minutes", "prefer_max_travel_minutes"):
            m = int(prm["minutes"])
            verb = "will travel" if t == "max_travel_minutes" else "prefers to travel"
            ok = lambda v: (c := travel.typical_minutes(p.base, v)) is not None and c <= m
            return self._venue_in(ok), f"{p.name} {verb} at most {m} min each way"

        venues = prm["venues"]
        names = ", ".join(sc.location_name(v) for v in venues)
        verb = "can only meet at" if t == "allowed_venues" else "prefers to meet at"
        return self._venue_in(lambda v: v in venues), f"{p.name} {verb} {names}"

    # ---------------------------------------------------------------- queries

    def _venue_index(self, venue: str | None) -> int | None:
        if not self.geo:
            return None
        if venue is None:
            raise ValueError("This meeting has venues; a proposal must name one")
        if venue not in self.venues:
            raise ValueError(f"'{venue}' is not a candidate venue ({', '.join(self.venues)})")
        return self.venues.index(venue)

    def _holds(self, c: Constraint, start: int, vi: int | None) -> bool:
        """Evaluate a constraint's formula at S = start (and V = vi)."""
        subs = [(self.S, z3.IntVal(start))]
        if vi is not None:
            subs.append((self.V, z3.IntVal(vi)))
        return z3.is_true(z3.simplify(z3.substitute(c.expr, *subs)))

    def verify(self, start: int, venue: str | None = None) -> Verdict:
        """Check a proposed start time (and venue) against every hard rule and preference."""
        vi = self._venue_index(venue)
        violations = [c for c in self.hard if not self._holds(c, start, vi)]
        met = [c for c in self.soft if self._holds(c, start, vi)]
        missed = [c for c in self.soft if c not in met]
        travel = {pid: costs[vi] for pid, costs in self.cost.items()} if self.geo else {}
        return Verdict(start, venue, not violations, violations, met, missed, travel)

    def _travel_objective(self, opt: z3.Optimize) -> None:
        mode = self.sc.meeting.travel_objective
        if not self.geo or mode == "none":
            return
        per_person = [z3.Sum([z3.If(self.V == i, c, 0) for i, c in enumerate(costs)])
                      for costs in self.cost.values()]
        if mode == "max":
            worst = z3.Int("max_travel")
            opt.add([worst >= x for x in per_person])
            opt.minimize(worst)
        else:
            opt.minimize(z3.Sum(per_person))

    def suggest(self, k: int = 3, min_gap: int | None = None) -> list[Option]:
        """Best k valid (slot, venue) options, ranked by preference score, then travel, then time.

        Options at the same venue are at least `min_gap` minutes apart (default:
        the meeting duration) so the list is not 15-minute shifts of one slot.
        """
        gap = self.D if min_gap is None else min_gap
        opt = z3.Optimize()
        for c in self.hard:
            opt.add(c.expr)
        for c in self.soft:
            opt.add_soft(c.expr, c.weight)
        self._travel_objective(opt)  # lower priority than soft goals
        opt.minimize(self.S)         # final tie-break: earliest

        options = []
        for _ in range(k):
            if opt.check() != z3.sat:
                break
            m = opt.model()
            start = m[self.S].as_long()
            vi = m.eval(self.V, model_completion=True).as_long() if self.geo else None
            venue = self.venues[vi] if self.geo else None
            options.append(Option(start, venue, self.verify(start, venue)))
            apart = z3.Or(self.S <= start - gap, self.S >= start + gap)
            opt.add(z3.Or(apart, self.V != vi) if self.geo else apart)
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

    def trip(self, p: Person, start: int, venue: str) -> Trip:
        """The journey a person would make for this proposal.

        `there` is the earliest possible arrival (used to explain violations).
        `plan` leaves as late as possible while still arriving by the start.
        """
        gap = next((g for g in self.gaps[p.id] if g.lo <= start and start + self.D <= g.hi), None)
        if gap is None:
            return Trip(p, None, None, None)
        travel = self.sc.travel
        plan = travel.latest_departure(gap.prev.location, venue, start)
        if plan is not None and plan.depart < gap.prev.end:
            plan = None
        after = travel.earliest_arrival(venue, gap.next.location, start + self.D) if gap.next else None
        if after is not None and after.arrive > gap.next.start:
            after = None
        return Trip(p, gap, self._route_there(gap, venue), self._route_onward(gap, venue), plan, after)

    def describe_route(self, r: Route, tz=None) -> str:
        sc = self.sc
        if r.mode == "none":
            return "no travel needed"
        parts = []
        for leg in r.legs:
            if leg.kind == "flight":
                dep, arr = sc.to_datetime(leg.start, tz), sc.to_datetime(leg.end, tz)
                arr_text = f"{arr:%H:%M}" if arr.date() == dep.date() else sc.fmt(leg.end, tz)
                parts.append(f"flight {leg.label} {sc.fmt(leg.start, tz)} -> {arr_text}")
            else:
                parts.append(f"{leg.end - leg.start} min by road to {leg.dest}")
        return (f"leave {sc.fmt(r.depart, tz)}, " + ", ".join(parts)
                + f", arrive {sc.fmt(r.arrive, tz)}")

    def _explain_trip(self, p: Person, start: int, vi: int | None, part: str) -> str:
        sc, venue = self.sc, self.venues[vi]
        t = self.trip(p, start, venue)
        vname = sc.location_name(venue)
        if t.gap is None:
            return f"{p.name} has no free gap around this time"
        if part == "there":
            if t.there is None:
                return (f"{p.name} cannot reach {vname} by {sc.fmt(start, p.tz)}: after {t.gap.prev.label} "
                        f"(free from {sc.fmt(t.gap.prev.end, p.tz)}) no flight gets there in time")
            return (f"{p.name} cannot reach {vname} by {sc.fmt(start, p.tz)}: after {t.gap.prev.label} "
                    f"(free from {sc.fmt(t.gap.prev.end, p.tz)}) the earliest arrival is "
                    f"{sc.fmt(t.there.arrive, p.tz)} ({self.describe_route(t.there, p.tz)})")
        end = start + self.D
        if t.onward is None:
            return (f"{p.name} cannot leave {vname} at {sc.fmt(end, p.tz)} and still make {t.gap.next.label}: "
                    f"no flight gets there in time")
        return (f"{p.name} would have to leave {vname} by {sc.fmt(t.onward.depart, p.tz)} to make "
                f"{t.gap.next.label}, but the meeting ends at {sc.fmt(end, p.tz)}")

    def explain(self, verdict: Verdict, public: bool = False) -> str:
        """Plain-language explanation of a verdict.

        With public=True (what negotiating agents see) only the hard-constraint
        outcome is reported; private preferences and travel plans are left out.
        """
        sc = self.sc
        slot = f"{sc.fmt(verdict.start)}-{sc.to_datetime(verdict.start + self.D):%H:%M} ({sc.display_tz.key})"
        vi = self._venue_index(verdict.venue)
        if verdict.venue:
            slot += f" at {sc.location_name(verdict.venue)}"
        lines = []
        if verdict.valid:
            lines.append(f"VALID: {slot} satisfies every hard constraint.")
        else:
            lines.append(f"INVALID: {slot} breaks {len(verdict.violations)} hard constraint(s):")
            lines += [f"  - {c.detail(verdict.start, vi) if c.detail else c.text}" for c in verdict.violations]
        if public:
            return "\n".join(lines)
        if verdict.valid and self.geo:
            lines.append("Travel plan:")
            for p in sc.participants():
                t = self.trip(p, verdict.start, verdict.venue)
                route = self.describe_route(t.plan, p.tz) if t.plan else "no travel computed"
                lines.append(f"  * {p.name}: {route}")
                if t.after and t.after.mode != "none":
                    lines.append(f"      then, for {t.gap.next.label}: {self.describe_route(t.after, p.tz)}")
            lines.append(f"Total one-way travel: {sum(verdict.travel.values())} min")
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
