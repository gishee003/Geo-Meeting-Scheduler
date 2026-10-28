from pathlib import Path

import pytest

from scheduler.models import load_scenario, scenario_from_dict
from scheduler.solver import MeetingSolver

SCENARIOS = Path(__file__).resolve().parent.parent / "data" / "scenarios"


def solver_for(name: str) -> MeetingSolver:
    return MeetingSolver(load_scenario(SCENARIOS / name))


def at(solver: MeetingSolver, text: str) -> int:
    sc = solver.sc
    return sc.parse_local(text, sc.display_tz)


def brute_force(solver: MeetingSolver) -> dict:
    """All valid (start, venue) -> verdict, by checking every slot and venue directly."""
    sc = solver.sc
    out = {}
    for start in range(0, sc.horizon_end, sc.slot_minutes):
        for venue in solver.venues or [None]:
            v = solver.verify(start, venue)
            if v.valid:
                out[(start, venue)] = v
    return out


def rank(solver: MeetingSolver, v) -> tuple:
    """The solver's lexicographic objective: preferences, then travel, then earliest."""
    mode = solver.sc.meeting.travel_objective
    travel = 0
    if v.travel and mode == "total":
        travel = sum(v.travel.values())
    elif v.travel and mode == "max":
        travel = max(v.travel.values())
    return (-v.score, travel, v.start)


# --- data model -----------------------------------------------------------------

def test_horizon_starts_at_local_midnight():
    s = solver_for("01_simple_team_sync.json")
    assert at(s, "2026-11-02T00:00") == 0
    assert at(s, "2026-11-03T10:00") == 24 * 60 + 600
    assert s.sc.fmt(0) == "Mon 02 Nov 00:00"


def test_unknown_participant_rejected():
    with pytest.raises(ValueError):
        scenario_from_dict(_mini(participants=["a", "ghost"]))


# --- scenario 1: simple ----------------------------------------------------------

def test_simple_best_slot_meets_all_preferences():
    s = solver_for("01_simple_team_sync.json")
    best = s.suggest(1)[0]
    assert best.start == at(s, "2026-11-04T11:00")
    assert best.verdict.valid and not best.verdict.prefs_missed


def test_verify_reports_busy_conflict():
    s = solver_for("01_simple_team_sync.json")
    v = s.verify(at(s, "2026-11-03T10:00"))
    assert not v.valid
    assert [c.id for c in v.violations] == ["asha:busy:0"]
    assert "Design review" in s.explain(v)


def test_verify_reports_day_rule():
    s = solver_for("01_simple_team_sync.json")
    v = s.verify(at(s, "2026-11-02T10:00"))
    assert [c.id for c in v.violations] == ["asha:hard:0"]


def test_off_grid_start_is_invalid():
    s = solver_for("01_simple_team_sync.json")
    v = s.verify(at(s, "2026-11-04T11:05"))
    assert "slot_grid" in [c.id for c in v.violations]


# --- scenario 2: trade-offs ----------------------------------------------------

def test_busy_week_trades_off_preferences():
    s = solver_for("02_busy_week.json")
    opts = s.suggest(5)
    assert [o.start for o in opts] == [at(s, "2026-11-04T14:00"), at(s, "2026-11-03T11:00")]
    assert opts[0].verdict.score == 8
    assert opts[0].verdict.score > opts[1].verdict.score


# --- scenario 3: impossible ----------------------------------------------------

def test_impossible_scenario_is_diagnosed():
    s = solver_for("03_impossible.json")
    diag = s.diagnose()
    assert not diag.feasible
    assert {c.id for c in diag.conflict} == {"rahul:hard:0", "priya:hard:0"}
    assert s.suggest(3) == []


# --- solver agrees with brute force ----------------------------------------------

@pytest.mark.parametrize("name", sorted(p.name for p in SCENARIOS.glob("*.json")))
def test_solver_matches_brute_force(name):
    s = solver_for(name)
    truth = brute_force(s)
    assert s.diagnose().feasible == bool(truth)
    opts = s.suggest(1)
    if truth:
        assert (opts[0].start, opts[0].venue) in truth
        assert rank(s, opts[0].verdict) == min(rank(s, v) for v in truth.values())
    else:
        assert opts == []


# --- edge cases ------------------------------------------------------------------

def _mini(participants=("a", "b"), **overrides):
    data = {
        "name": "mini",
        "timezone": "Asia/Kolkata",
        "horizon": {"start": "2026-11-02", "days": 1},
        "meeting": {"title": "m", "duration_minutes": 60, "participants": list(participants)},
        "people": [
            {"id": "a", "working_hours": {"start": "09:00", "end": "12:00"},
             "calendar": [{"start": "2026-11-02T09:00", "end": "2026-11-02T10:00"}]},
            {"id": "b", "working_hours": {"start": "09:00", "end": "12:00"}},
        ],
    }
    data.update(overrides)
    return data


def test_back_to_back_meeting_is_allowed():
    s = MeetingSolver(scenario_from_dict(_mini()))
    assert s.verify(at(s, "2026-11-02T10:00")).valid
    assert not s.verify(at(s, "2026-11-02T09:45")).valid


def test_meeting_must_fit_inside_working_hours():
    s = MeetingSolver(scenario_from_dict(_mini()))
    assert s.verify(at(s, "2026-11-02T11:00")).valid
    assert not s.verify(at(s, "2026-11-02T11:15")).valid


def test_working_hours_use_each_persons_time_zone():
    # b works 09:00-12:00 London = 14:30-17:30 IST (GMT in November).
    data = _mini()
    data["people"][1]["timezone"] = "Europe/London"
    s = MeetingSolver(scenario_from_dict(data))
    assert s.diagnose().feasible is False  # a's 09:00-12:00 IST never meets b's window

    data["people"][0]["working_hours"]["end"] = "18:00"
    s = MeetingSolver(scenario_from_dict(data))
    assert s.suggest(1)[0].start == at(s, "2026-11-02T14:30")
