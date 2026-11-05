import matplotlib

matplotlib.use("Agg")

import pytest
from streamlit.testing.v1 import AppTest

from scheduler import ui

APP = str(ui.ROOT / "app.py")


def test_scenarios_listed_with_tiers():
    stems = [s for s, _, _ in ui.list_scenarios()]
    assert stems[0] == "01_simple_team_sync" and "09_plain_language_notes" in stems
    assert all(tier for _, _, tier in ui.list_scenarios())


def test_people_rows_show_rules_and_notes():
    rows = ui.people_rows(ui.load("09_plain_language_notes"))
    priya = next(r for r in rows if r["Person"] == "Priya")
    assert "pick up my kids" in priya["Notes (plain English)"]
    rows = ui.people_rows(ui.load("05_multi_city_flights"))
    asha = next(r for r in rows if r["Person"] == "Asha")
    assert "in person" in asha["Must-follow rules"] and asha["Based at"].startswith("Bengaluru")


def test_final_check_uses_what_people_really_need():
    sc = ui.load("09_plain_language_notes")
    tue_3pm = sc.parse_local("2026-11-03T15:00", sc.display_tz)   # breaks Priya's school run
    check = ui.final_check(sc, tue_3pm, None)
    assert not check.verdict.valid and "15:00 and 18:00" in check.explanation
    assert check.best is not None and check.max_score == 7


def test_final_check_without_a_slot():
    check = ui.final_check(ui.load("03_impossible"), None, None)
    assert check.verdict is None and check.best is None


def test_saved_runs_replay():
    paths = ui.saved_runs("06_london_time_zones", "llm_only")
    assert len(paths) == 3
    row, entries = ui.load_saved(paths[0])
    assert row["system"] == "llm_only" and entries[0].speaker == "Mediator"


def test_phase4_summary_for_a_scenario():
    summary = {r["System"]: r for r in ui.phase4_summary("06_london_time_zones")}
    assert summary[ui.MODES["hybrid"]]["Correct"] == "3/3"
    assert summary[ui.MODES["llm_only"]]["Impossible meeting scheduled"] == 3


@pytest.mark.parametrize("stem", ["01_simple_team_sync", "05_multi_city_flights", "08_global_five_people"])
def test_figures_render(stem):
    sc = ui.load(stem)
    best = ui.final_check(sc, None, None).best
    assert ui.calendar_figure(sc, best.start, best.venue) is not None
    assert (ui.map_figure(sc, best.venue) is None) == (not sc.geo_enabled)


def test_heuristic_live_run_streams_messages():
    posted = []
    out, llm = ui.run_live(ui.load("04_same_city_travel"), "heuristic", [], 8, posted.append)
    assert out.status == "agreed" and llm is None and posted[0].speaker == "Mediator"


def test_live_llm_run_needs_a_provider(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No AI provider"):
        ui.make_llm(["gemini:x"])


# --- the app itself -----------------------------------------------------------------------

def test_app_loads_without_errors():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Simple team sync"


def test_app_replays_a_saved_run():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.selectbox[0].set_value("06_london_time_zones").run()
    at.sidebar.radio[0].set_value("llm_only").run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert any("Invalid meeting" in e.value for e in at.error)    # AI alone booked 04:30 London time


def test_app_replays_hybrid_on_plain_language_scenario():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.selectbox[0].set_value("09_plain_language_notes").run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert any("Valid meeting" in s.value for s in at.success)
