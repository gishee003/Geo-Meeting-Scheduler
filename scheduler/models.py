"""Data model for a scheduling scenario.

All times are converted to integer "minutes since the start of the planning
horizon" (measured in UTC). The solver only works with these integers, which
keeps the Z3 encoding simple and makes time zones a data-loading concern.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .geo import TravelModel, load_geo

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass
class Event:
    """An existing calendar entry. start/end are horizon minutes."""
    title: str
    start: int
    end: int
    location: str | None = None  # location id; None means the person's base


@dataclass
class WorkingHours:
    start: time
    end: time
    days: list[str]


@dataclass
class Rule:
    """A hard constraint or soft preference, kept as raw JSON fields.

    `type` selects the encoding in the solver; `params` holds the rest.
    Soft preferences additionally carry a `weight`.
    """
    type: str
    params: dict
    weight: int = 0


@dataclass
class Person:
    id: str
    name: str
    tz: ZoneInfo
    working_hours: WorkingHours
    calendar: list[Event] = field(default_factory=list)
    hard_constraints: list[Rule] = field(default_factory=list)
    preferences: list[Rule] = field(default_factory=list)
    # Geography (Phase 2). Each day the person starts at `base` no earlier than
    # `earliest_departure` and must be back at `base` by `return_by` (local times).
    base: str | None = None
    earliest_departure: time = time(6, 0)
    return_by: time | None = time(23, 0)


@dataclass
class Meeting:
    title: str
    duration: int  # minutes
    participants: list[str]
    venues: list[str] = field(default_factory=list)  # candidate location ids; empty = no geography
    travel_objective: str = "total"                   # "total", "max" or "none"


@dataclass
class Scenario:
    name: str
    description: str
    horizon_start: datetime  # aware datetime (UTC)
    horizon_days: int
    slot_minutes: int
    meeting: Meeting
    people: dict[str, Person]
    display_tz: ZoneInfo
    travel: TravelModel | None = None

    @property
    def geo_enabled(self) -> bool:
        return bool(self.meeting.venues)

    def location_name(self, loc_id: str) -> str:
        return self.travel.loc[loc_id].name if self.travel else loc_id

    @property
    def horizon_end(self) -> int:
        return self.horizon_days * 24 * 60

    def participants(self) -> list[Person]:
        return [self.people[pid] for pid in self.meeting.participants]

    # --- time conversion helpers -------------------------------------------

    def to_minutes(self, dt: datetime) -> int:
        return int((dt.astimezone(timezone.utc) - self.horizon_start).total_seconds() // 60)

    def to_datetime(self, minutes: int, tz: ZoneInfo | None = None) -> datetime:
        dt = self.horizon_start + timedelta(minutes=minutes)
        return dt.astimezone(tz or self.display_tz)

    def parse_local(self, text: str, tz: ZoneInfo) -> int:
        """Parse 'YYYY-MM-DDTHH:MM' in the given zone into horizon minutes."""
        return self.to_minutes(datetime.fromisoformat(text).replace(tzinfo=tz))

    def fmt(self, minutes: int, tz: ZoneInfo | None = None) -> str:
        return self.to_datetime(minutes, tz).strftime("%a %d %b %H:%M")

    def local_days(self, tz: ZoneInfo) -> list[tuple[date, int]]:
        """Each local calendar day overlapping the horizon, with its midnight in horizon minutes.

        One extra day on each side covers time-zone offsets.
        """
        first = self.horizon_start.astimezone(tz).date() - timedelta(days=1)
        days = []
        for i in range(self.horizon_days + 2):
            d = first + timedelta(days=i)
            midnight = self.to_minutes(datetime.combine(d, time(0, 0), tzinfo=tz))
            days.append((d, midnight))
        return days

    def local_window(self, d: date, start: time, end: time, tz: ZoneInfo) -> tuple[int, int]:
        """Horizon-minute interval for a local clock window on day d."""
        s = self.to_minutes(datetime.combine(d, start, tzinfo=tz))
        e = self.to_minutes(datetime.combine(d, end, tzinfo=tz))
        return s, e


# --- loading -----------------------------------------------------------------

def _parse_time(text: str) -> time:
    return time.fromisoformat(text)


def _parse_rule(raw: dict, soft: bool) -> Rule:
    raw = dict(raw)
    rule_type = raw.pop("type")
    weight = int(raw.pop("weight", 1)) if soft else 0
    return Rule(type=rule_type, params=raw, weight=weight)


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    return scenario_from_dict(json.loads(path.read_text(encoding="utf-8")), base_dir=path.parent)


def scenario_from_dict(data: dict, base_dir: str | Path = ".") -> Scenario:
    display_tz = ZoneInfo(data.get("timezone", "UTC"))
    horizon = data["horizon"]
    start_day = date.fromisoformat(horizon["start"])
    horizon_start = datetime.combine(start_day, time(0, 0), tzinfo=display_tz).astimezone(timezone.utc)

    m = data["meeting"]
    meeting = Meeting(
        title=m["title"],
        duration=int(m["duration_minutes"]),
        participants=list(m["participants"]),
        venues=list(m.get("venues", [])),
        travel_objective=m.get("travel_objective", "total"),
    )

    scenario = Scenario(
        name=data["name"],
        description=data.get("description", ""),
        horizon_start=horizon_start,
        horizon_days=int(horizon["days"]),
        slot_minutes=int(data.get("slot_minutes", 15)),
        meeting=meeting,
        people={},
        display_tz=display_tz,
    )
    if "geo" in data:
        geo = load_geo(Path(base_dir) / data["geo"])
        scenario.travel = TravelModel(geo, horizon_start, scenario.horizon_days)

    for p in data["people"]:
        tz = ZoneInfo(p.get("timezone", data.get("timezone", "UTC")))
        wh = p.get("working_hours", {})
        working_hours = WorkingHours(
            start=_parse_time(wh.get("start", "09:00")),
            end=_parse_time(wh.get("end", "17:00")),
            days=list(wh.get("days", DAY_NAMES[:5])),
        )
        calendar = [
            Event(
                title=e.get("title", "Busy"),
                start=scenario.parse_local(e["start"], tz),
                end=scenario.parse_local(e["end"], tz),
                location=e.get("location"),
            )
            for e in p.get("calendar", [])
        ]
        return_by = p.get("return_by", "23:00")
        scenario.people[p["id"]] = Person(
            id=p["id"],
            name=p.get("name", p["id"]),
            tz=tz,
            working_hours=working_hours,
            calendar=calendar,
            hard_constraints=[_parse_rule(r, soft=False) for r in p.get("hard_constraints", [])],
            preferences=[_parse_rule(r, soft=True) for r in p.get("preferences", [])],
            base=p.get("base"),
            earliest_departure=_parse_time(p.get("earliest_departure", "06:00")),
            return_by=_parse_time(return_by) if return_by else None,
        )

    missing = [pid for pid in meeting.participants if pid not in scenario.people]
    if missing:
        raise ValueError(f"Unknown participants in meeting: {missing}")
    if scenario.geo_enabled:
        _check_geo(scenario)
    return scenario


def _check_geo(sc: Scenario) -> None:
    if sc.travel is None:
        raise ValueError("Meeting lists venues but the scenario has no 'geo' file")
    known = sc.travel.loc
    used = list(sc.meeting.venues)
    for p in sc.participants():
        if p.base is None:
            raise ValueError(f"{p.id} needs a 'base' location when the meeting has venues")
        used.append(p.base)
        used += [e.location for e in p.calendar if e.location]
    unknown = sorted({u for u in used if u not in known})
    if unknown:
        raise ValueError(f"Unknown location ids: {unknown}")
