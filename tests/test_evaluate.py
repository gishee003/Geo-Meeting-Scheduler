from pathlib import Path

import pytest

from scheduler.evaluate import (LLM_SYSTEMS, Truth, TIERS, ground_truth, is_correct, is_unsafe, make_row, plan,
                                run_solver_only)
from scheduler.models import load_scenario, with_truth
from scheduler.negotiation import Outcome
from scheduler.solver import MeetingSolver
from scheduler.translate import translate_scenario

SCENARIOS = Path(__file__).resolve().parent.parent / "data" / "scenarios"
ALL = sorted(p.stem for p in SCENARIOS.glob("*.json"))


def solver_for(name):
    return MeetingSolver(load_scenario(SCENARIOS / f"{name}.json"))


@pytest.mark.parametrize("status,valid,feasible,expected", [
    ("agreed", True, True, True),
    ("fallback", True, True, True),
    ("solved", True, True, True),
    ("agreed", False, True, False),       # scheduled something impossible
    ("no_agreement", False, True, False),  # gave up although a slot exists
    ("impossible", False, False, True),   # correctly proved
    ("no_agreement", False, False, False),  # gave up, but never proved it
    ("agreed", False, False, False),
])
def test_is_correct(status, valid, feasible, expected):
    assert is_correct(status, valid, feasible) is expected


def test_is_unsafe_only_for_agreed_invalid_slots():
    assert is_unsafe("agreed", False)
    assert not is_unsafe("agreed", True)
    assert not is_unsafe("no_agreement", False)
    assert not is_unsafe("fallback", True)


def test_every_scenario_has_a_tier():
    assert set(ALL) == set(TIERS)


def test_only_scenario_03_is_impossible():
    assert [n for n in ALL if not ground_truth(solver_for(n)).feasible] == ["03_impossible"]


def test_plan_repeats_only_llm_systems():
    jobs = plan(["a", "b"], ["llm_only", "solver_only", "hybrid"], 3)
    assert len(jobs) == 2 * (3 * len(LLM_SYSTEMS) + 1)
    assert ("a", "solver_only", 2) not in jobs
    assert ("b", "hybrid", 3) in jobs
    assert [j[2] for j in jobs[:3]] == [1, 1, 1]  # first repeat sweeps everything first


def test_solver_only_is_correct_everywhere():
    for name in ALL:
        s = solver_for(name)
        truth = ground_truth(s)
        row = make_row(name, "solver_only", 1, truth, run_solver_only(s), 0.1)
        assert row["correct"], name
        assert row["llm_calls"] == 0


def test_make_row_records_a_failed_llm_agreement():
    s = solver_for("01_simple_team_sync")
    start = s.sc.parse_local("2026-11-02T10:00", s.sc.display_tz)   # Asha has no meetings on Monday
    out = Outcome("agreed", start, None, s.verify(start))
    row = make_row("01_simple_team_sync", "llm_only", 2, ground_truth(s), out, 3.0, (5, 100, 20), "m")
    assert not row["correct"] and row["unsafe"] and not row["valid"]
    assert (row["llm_calls"], row["tier"], row["best_score"]) == (5, "easy", 7)


def test_no_verdict_means_no_score():
    row = make_row("x", "llm_only", 1, Truth(True, 3, 10, 5), Outcome("no_agreement"), 1.0)
    assert row["score"] is None and row["travel"] is None and not row["correct"]


# --- plain-language notes ---------------------------------------------------------

class TranslatorLLM:
    """Stands in for the LLM: returns fixed translations, one reply per person with notes."""

    def __init__(self, replies):
        self.replies = list(replies)

    def json(self, system, prompt, schema):
        return {"rules": self.replies.pop(0)}


def rule(kind, strength, start="", end="", days=(), importance=1, note="n"):
    return {"note": note, "kind": kind, "strength": strength, "start": start, "end": end,
            "days": list(days), "importance": importance}


NOTES = "09_plain_language_notes"


def test_solver_alone_ignores_plain_language_notes():
    sc = load_scenario(SCENARIOS / f"{NOTES}.json")
    best = MeetingSolver(sc).suggest(1)[0]
    assert not MeetingSolver(with_truth(sc)).verify(best.start).valid


def test_translated_notes_become_enforced_rules():
    sc = load_scenario(SCENARIOS / f"{NOTES}.json")
    llm = TranslatorLLM([
        [rule("avoid_days", "hard", days=["Wed"])],
        [rule("avoid_days", "soft", days=["Mon"], importance=2)],
        [rule("blocked_window", "hard", "15:00", "18:00", ["Tue", "Thu"])],
    ])
    translated, records = translate_scenario(sc, llm)
    assert len(records) == 3 and all(r.rule for r in records)
    best = MeetingSolver(translated).suggest(1)[0]
    truth = MeetingSolver(with_truth(sc))
    assert truth.verify(best.start).valid
    assert best.verdict.score == ground_truth(truth).best_score


def test_malformed_translation_is_dropped_and_reported():
    sc = load_scenario(SCENARIOS / f"{NOTES}.json")
    llm = TranslatorLLM([[rule("blocked_window", "hard", "late", "later")], [], []])
    translated, records = translate_scenario(sc, llm)
    assert records[0].rule is None
    assert "Could not use" in records[0].summary()
    assert translated.people["asha"].hard_constraints == []


def test_soft_prefer_days_cannot_become_hard():
    sc = load_scenario(SCENARIOS / f"{NOTES}.json")
    llm = TranslatorLLM([[rule("prefer_days", "hard", days=["Fri"], importance=2)], [], []])
    _, records = translate_scenario(sc, llm)
    assert records[0].rule.type == "prefer_days" and records[0].rule.weight == 2


def test_notes_reach_llm_agents_but_truth_does_not():
    from scheduler.agents import LLMAgent
    sc = load_scenario(SCENARIOS / f"{NOTES}.json")
    asha = LLMAgent(sc.people["asha"], sc, TranslatorLLM([]))
    assert "family wedding" in asha.system
    assert "no_meetings_on_days" not in asha.system


# --- scenario 10: both single approaches fail -------------------------------------------

BOTH = "10_notes_and_time_zones"


def test_scenario_10_has_exactly_one_valid_slot():
    sc = load_scenario(SCENARIOS / f"{BOTH}.json")
    truth = MeetingSolver(with_truth(sc))
    valid = [(t, v) for t in range(0, sc.horizon_end, sc.slot_minutes) for v in sc.meeting.venues
             if truth.verify(t, v).valid]
    assert valid == [(sc.parse_local("2026-11-02T16:00", sc.display_tz), "online")]


def test_scenario_10_solver_alone_breaks_the_note():
    sc = load_scenario(SCENARIOS / f"{BOTH}.json")
    best = MeetingSolver(sc).suggest(1)[0]
    verdict = MeetingSolver(with_truth(sc)).verify(best.start, best.venue)
    assert not verdict.valid
    assert [c.id for c in verdict.violations] == ["asha:hard:0"]   # the physiotherapy note


def test_scenario_10_india_morning_is_dawn_in_london():
    sc = load_scenario(SCENARIOS / f"{BOTH}.json")
    v = MeetingSolver(with_truth(sc)).verify(sc.parse_local("2026-11-03T10:00", sc.display_tz), "online")
    assert [c.id for c in v.violations] == ["emma:working_hours"]


def test_scenario_10_translated_note_leads_to_the_only_valid_slot():
    sc = load_scenario(SCENARIOS / f"{BOTH}.json")
    translated, _ = translate_scenario(sc, TranslatorLLM([[rule("blocked_window", "hard", "15:00", "18:00",
                                                                ["Tue", "Wed"])]]))
    best = MeetingSolver(translated).suggest(1)[0]
    assert (best.start, best.venue) == (sc.parse_local("2026-11-02T16:00", sc.display_tz), "online")


def test_day_limited_only_between_keeps_its_days():
    # Regression: "only available 00:00-15:00 on Tue, Wed" once lost its days and blocked every afternoon.
    sc = load_scenario(SCENARIOS / f"{BOTH}.json")
    translated, records = translate_scenario(sc, TranslatorLLM([[rule("only_between", "hard", "00:00", "15:00",
                                                                     ["Tue", "Wed"])]]))
    assert records[0].rule.params["days"] == ["Tue", "Wed"]
    assert "on Tue, Wed" in records[0].text
    s = MeetingSolver(translated)
    assert s.verify(sc.parse_local("2026-11-02T16:00", sc.display_tz), "online").valid       # Monday is free
    assert not s.verify(sc.parse_local("2026-11-03T15:30", sc.display_tz), "online").valid   # Tuesday is not
    best = s.suggest(1)[0]
    assert (best.start, best.venue) == (sc.parse_local("2026-11-02T16:00", sc.display_tz), "online")
