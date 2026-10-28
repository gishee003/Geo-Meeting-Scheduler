import json
from pathlib import Path

import pytest

from scheduler.models import load_scenario, scenario_from_dict
from scheduler.solver import MeetingSolver

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ROOT / "data" / "scenarios"


def solver_for(name: str) -> MeetingSolver:
    return MeetingSolver(load_scenario(SCENARIOS / name))


def at(solver: MeetingSolver, text: str, tz=None) -> int:
    sc = solver.sc
    return sc.parse_local(text, tz or sc.display_tz)


def raw(name: str) -> dict:
    return json.loads((SCENARIOS / name).read_text(encoding="utf-8"))


# --- travel model -----------------------------------------------------------------

@pytest.fixture(scope="module")
def flights():
    return solver_for("05_multi_city_flights.json")


def test_ground_time_within_city(flights):
    t = flights.sc.travel
    assert t.ground_minutes("blr_ec_office", "blr_kor_office") == 50
    assert t.ground_minutes("blr_ec_office", "online") == 0
    assert not t.needs_flight("blr_ec_office", "blr_kor_office")


def test_long_distances_need_a_flight(flights):
    t = flights.sc.travel
    assert t.needs_flight("blr_ec_office", "bom_office")
    assert t.needs_flight("del_office", "lon_office")


def test_earliest_arrival_uses_timetable(flights):
    t = flights.sc.travel
    ready = at(flights, "2026-11-03T06:00")
    r = t.earliest_arrival("blr_ec_office", "bom_office", ready)
    # 100 min to BLR + 60 min check-in rules out 06:00 and 08:30; first usable is MR101 at 11:00.
    flight = next(l for l in r.legs if l.kind == "flight")
    assert flight.label == "MR101" and flight.start == at(flights, "2026-11-03T11:00")
    assert r.arrive == at(flights, "2026-11-03T13:25")


def test_latest_departure_is_mirror_of_earliest_arrival(flights):
    t = flights.sc.travel
    r = t.latest_departure("bom_office", "blr_ec_office", at(flights, "2026-11-03T23:00"))
    flight = next(l for l in r.legs if l.kind == "flight")
    assert flight.start == at(flights, "2026-11-03T18:30")
    assert r.depart == at(flights, "2026-11-03T17:10")


def test_flight_arrival_respects_destination_time_zone():
    s = solver_for("06_london_time_zones.json")
    t, sc = s.sc.travel, s.sc
    lhr = t.loc["LHR"].tz
    r = t.earliest_arrival("BLR", "LHR", at(s, "2026-11-03T04:00"))
    flight = next(l for l in r.legs if l.kind == "flight")
    # Departs 07:00 IST, 10h30 in the air, lands 12:00 GMT.
    assert sc.to_datetime(flight.start).strftime("%H:%M") == "07:00"
    assert sc.to_datetime(flight.end, lhr).strftime("%H:%M") == "12:00"


# --- scenario 4: same city -----------------------------------------------------

def test_calendar_free_slot_rejected_for_travel():
    s = solver_for("04_same_city_travel.json")
    v = s.verify(at(s, "2026-11-02T11:00"), "blr_ec_office")
    assert not v.valid
    assert [c.id for c in v.violations] == ["asha:reach_venue"]
    assert "11:50" in s.explain(v)


def test_same_city_picks_venue_that_avoids_travel_clash():
    s = solver_for("04_same_city_travel.json")
    best = s.suggest(1)[0]
    assert (best.start, best.venue) == (at(s, "2026-11-02T11:00"), "blr_kor_office")


def test_onward_travel_to_next_meeting_is_enforced():
    s = solver_for("04_same_city_travel.json")
    # Rahul's 13:00 vendor visit is at Electronic City, 50 min from Koramangala.
    v = s.verify(at(s, "2026-11-02T12:00"), "blr_kor_office")
    assert [c.id for c in v.violations] == ["rahul:onward_travel"]


# --- scenario 5: flights --------------------------------------------------------

def test_flight_timetable_rules_out_early_slot():
    s = solver_for("05_multi_city_flights.json")
    v = s.verify(at(s, "2026-11-03T10:00"), "bom_office")
    assert {c.id for c in v.violations} == {"asha:reach_venue", "priya:reach_venue"}


def test_multi_city_best_option():
    s = solver_for("05_multi_city_flights.json")
    best = s.suggest(1)[0]
    assert (best.start, best.venue) == (at(s, "2026-11-03T14:00"), "bom_office")
    assert best.verdict.travel == {"asha": 305, "rahul": 0, "priya": 265}


def test_same_day_return_is_required():
    s = solver_for("05_multi_city_flights.json")
    # Monday at Mumbai: Asha cannot fly back for Board prep at 15:00 in Bengaluru.
    v = s.verify(at(s, "2026-11-02T13:30"), "bom_office")
    assert "asha:onward_travel" in [c.id for c in v.violations]


def test_in_person_only_excludes_online():
    s = solver_for("05_multi_city_flights.json")
    v = s.verify(at(s, "2026-11-03T14:00"), "online")
    assert [c.id for c in v.violations] == ["asha:hard:0"]


# --- scenario 6: time zones -------------------------------------------------------

def test_london_meeting_goes_online_in_overlap_hours():
    s = solver_for("06_london_time_zones.json")
    best = s.suggest(1)[0]
    assert best.venue == "online"
    assert best.start == at(s, "2026-11-03T10:00", s.sc.people["emma"].tz)


def test_forcing_in_person_with_london_is_infeasible():
    data = raw("06_london_time_zones.json")
    data["people"][2]["hard_constraints"] = [{"type": "in_person_only"}]
    s = MeetingSolver(scenario_from_dict(data, base_dir=SCENARIOS))
    diag = s.diagnose()
    assert not diag.feasible
    ids = {c.id for c in diag.conflict}
    assert "emma:hard:0" in ids
    assert ids & {"asha:reach_venue", "asha:onward_travel", "rahul:reach_venue",
                  "rahul:onward_travel", "emma:reach_venue", "emma:onward_travel"}


# --- validation -----------------------------------------------------------------

def test_unknown_location_is_rejected():
    data = raw("04_same_city_travel.json")
    data["meeting"]["venues"].append("atlantis")
    with pytest.raises(ValueError, match="atlantis"):
        scenario_from_dict(data, base_dir=SCENARIOS)


def test_geo_proposal_must_name_a_venue():
    s = solver_for("04_same_city_travel.json")
    with pytest.raises(ValueError):
        s.verify(at(s, "2026-11-02T11:00"))
