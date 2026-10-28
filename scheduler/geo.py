"""Geography: locations, ground-travel estimates and a flight timetable.

Everything here is *data preparation* for the solver. Given a fixed starting
point and time (e.g. "Rahul is free at the Mumbai office from 06:00") the
travel model computes, in plain Python, the earliest time that person can be
at a venue, or the latest time they must leave a venue to make their next
commitment. These numbers become constants in the Z3 constraints, so the
solver itself stays linear integer arithmetic.

Simplifications (stated in the report):
  * ground time = road distance (great-circle x road factor) at a two-tier
    speed (city / highway) plus a fixed overhead;
  * trips longer than `max_drive_km` must fly; only direct flights are used;
  * flight schedules repeat every day of the horizon.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


@dataclass
class Location:
    id: str
    name: str
    kind: str                      # "office", "airport", "online", ...
    city: str | None = None
    country: str | None = None
    tz: ZoneInfo | None = None
    lat: float | None = None
    lon: float | None = None
    airport: str | None = None     # nearest airport id (non-airport locations)

    @property
    def online(self) -> bool:
        return self.kind == "online"


@dataclass
class Flight:
    number: str
    origin: str
    dest: str
    dep: int  # horizon minutes
    arr: int


@dataclass
class Leg:
    kind: str     # "ground" or "flight"
    origin: str   # location names, for explanations
    dest: str
    start: int
    end: int
    label: str = ""


@dataclass
class Route:
    depart: int
    arrive: int
    legs: list[Leg] = field(default_factory=list)

    @property
    def mode(self) -> str:
        if any(l.kind == "flight" for l in self.legs):
            return "flight"
        return "ground" if self.legs else "none"


@dataclass
class GeoData:
    locations: dict[str, Location]
    schedules: list[dict]
    params: dict


def load_geo(path: str | Path) -> GeoData:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    locations = {}
    for raw in data["locations"] + [dict(a, kind="airport") for a in data.get("airports", [])]:
        locations[raw["id"]] = Location(
            id=raw["id"],
            name=raw.get("name", raw["id"]),
            kind=raw.get("kind", "office"),
            city=raw.get("city"),
            country=raw.get("country"),
            tz=ZoneInfo(raw["timezone"]) if "timezone" in raw else None,
            lat=raw.get("lat"),
            lon=raw.get("lon"),
            airport=raw.get("airport"),
        )
    return GeoData(locations, data.get("flights", []), data["travel"])


def haversine_km(a: Location, b: Location) -> float:
    r = 6371.0
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dp, dl = p2 - p1, math.radians(b.lon - a.lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


class TravelModel:
    def __init__(self, geo: GeoData, horizon_start: datetime, horizon_days: int):
        self.geo = geo
        self.loc = geo.locations
        self.p = geo.params
        self.horizon_start = horizon_start
        self.flights: dict[tuple[str, str], list[Flight]] = {}
        self._expand_timetable(horizon_days)

    # ------------------------------------------------------------- timetable

    def _minutes(self, dt: datetime) -> int:
        return int((dt.astimezone(timezone.utc) - self.horizon_start).total_seconds() // 60)

    def _expand_timetable(self, horizon_days: int) -> None:
        """Turn daily schedules into concrete flights across the horizon (+1 day each side)."""
        for s in self.geo.schedules:
            tz = self.loc[s["from"]].tz
            first = self.horizon_start.astimezone(tz).date() - timedelta(days=1)
            for i in range(horizon_days + 2):
                d = first + timedelta(days=i)
                for hhmm in s["departures"]:
                    dep = datetime.combine(d, time.fromisoformat(hhmm), tzinfo=tz)
                    arr = dep + timedelta(minutes=s["duration_minutes"])
                    self.flights.setdefault((s["from"], s["to"]), []).append(
                        Flight(s["number"], s["from"], s["to"], self._minutes(dep), self._minutes(arr)))
        for fl in self.flights.values():
            fl.sort(key=lambda f: f.dep)

    # ---------------------------------------------------------------- ground

    def _road_km(self, a: Location, b: Location) -> float:
        return haversine_km(a, b) * self.p["road_factor"]

    def ground_minutes(self, a: str, b: str) -> int:
        A, B = self.loc[a], self.loc[b]
        if a == b or A.online or B.online:
            return 0
        km = self._road_km(A, B)
        thr = self.p["city_threshold_km"]
        hours = min(km, thr) / self.p["city_speed_kmh"] + max(0.0, km - thr) / self.p["highway_speed_kmh"]
        minutes = hours * 60 + self.p["ground_overhead_minutes"]
        step = self.p["round_to_minutes"]
        return int(math.ceil(minutes / step) * step)

    def needs_flight(self, a: str, b: str) -> bool:
        A, B = self.loc[a], self.loc[b]
        if a == b or A.online or B.online:
            return False
        return self._road_km(A, B) > self.p["max_drive_km"]

    # --------------------------------------------------------------- flights

    def _airport(self, a: str) -> str:
        A = self.loc[a]
        return a if A.kind == "airport" else A.airport

    def _checkin(self, ap_from: str, ap_to: str) -> int:
        international = self.loc[ap_from].country != self.loc[ap_to].country
        return self.p["international_checkin_minutes" if international else "checkin_minutes"]

    def _flight_routes(self, a: str, b: str):
        """Yield (leave_a, arrive_b, legs) for every direct flight connecting a to b."""
        ap_a, ap_b = self._airport(a), self._airport(b)
        g1, g2 = self.ground_minutes(a, ap_a), self.ground_minutes(ap_b, b)
        ci, dp = self._checkin(ap_a, ap_b), self.p["deplane_minutes"]
        A, B, PA, PB = self.loc[a], self.loc[b], self.loc[ap_a], self.loc[ap_b]
        for f in self.flights.get((ap_a, ap_b), []):
            leave, arrive = f.dep - ci - g1, f.arr + dp + g2
            legs = [Leg("ground", A.name, PA.name, leave, leave + g1),
                    Leg("flight", PA.name, PB.name, f.dep, f.arr, f.number),
                    Leg("ground", PB.name, B.name, f.arr + dp, arrive)]
            yield leave, arrive, [l for l in legs if l.kind == "flight" or l.end > l.start]

    def earliest_arrival(self, a: str, b: str, ready: int) -> Route | None:
        """Earliest time someone free at `a` from `ready` can be at `b`; None if impossible."""
        if not self.needs_flight(a, b):
            g = self.ground_minutes(a, b)
            legs = [Leg("ground", self.loc[a].name, self.loc[b].name, ready, ready + g)] if g else []
            return Route(ready, ready + g, legs)
        options = [(arr, lv, legs) for lv, arr, legs in self._flight_routes(a, b) if lv >= ready]
        if not options:
            return None
        arr, lv, legs = min(options, key=lambda o: o[0])
        return Route(lv, arr, legs)

    def latest_departure(self, a: str, b: str, deadline: int) -> Route | None:
        """Latest time someone can leave `a` and still be at `b` by `deadline`; None if impossible."""
        if not self.needs_flight(a, b):
            g = self.ground_minutes(a, b)
            legs = [Leg("ground", self.loc[a].name, self.loc[b].name, deadline - g, deadline)] if g else []
            return Route(deadline - g, deadline, legs)
        options = [(lv, arr, legs) for lv, arr, legs in self._flight_routes(a, b) if arr <= deadline]
        if not options:
            return None
        lv, arr, legs = max(options, key=lambda o: o[0])
        return Route(lv, arr, legs)

    def typical_minutes(self, a: str, b: str) -> int | None:
        """Door-to-door one-way travel time ignoring the timetable (used for the travel objective)."""
        if not self.needs_flight(a, b):
            return self.ground_minutes(a, b)
        ap_a, ap_b = self._airport(a), self._airport(b)
        flights = self.flights.get((ap_a, ap_b))
        if not flights:
            return None
        return (self.ground_minutes(a, ap_a) + self._checkin(ap_a, ap_b) + min(f.arr - f.dep for f in flights)
                + self.p["deplane_minutes"] + self.ground_minutes(ap_b, b))
