import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scheduler.agents import Agent, HeuristicAgent, LLMAgent
from scheduler.llm import ClaudeLLM
from scheduler.models import load_scenario
from scheduler.negotiation import Negotiation
from scheduler.protocol import Proposal, Reply

SCENARIOS = Path(__file__).resolve().parent.parent / "data" / "scenarios"


def scenario(name):
    return load_scenario(SCENARIOS / name)


def heuristic(sc):
    return [HeuristicAgent(p, sc) for p in sc.participants()]


# --- negotiation loop with offline agents ---------------------------------------

@pytest.mark.parametrize("name", ["01_simple_team_sync.json", "04_same_city_travel.json",
                                  "05_multi_city_flights.json", "06_london_time_zones.json"])
def test_heuristic_agents_reach_valid_agreement(name):
    sc = scenario(name)
    out = Negotiation(sc, heuristic(sc)).run()
    assert out.status == "agreed"
    assert out.valid


def test_impossible_scenario_stops_before_negotiating():
    sc = scenario("03_impossible.json")
    out = Negotiation(sc, heuristic(sc)).run()
    assert out.status == "impossible"
    assert out.proposals == 0
    assert "NO VALID SLOT" in out.transcript[-1].text


def test_referee_catches_invalid_proposals():
    sc = scenario("01_simple_team_sync.json")
    out = Negotiation(sc, heuristic(sc)).run()
    assert out.invalid_caught > 0
    assert any(e.speaker == "Referee" and e.text.startswith("INVALID") for e in out.transcript)


def test_round_limit_falls_back_to_solver():
    sc = scenario("01_simple_team_sync.json")
    out = Negotiation(sc, heuristic(sc), max_rounds=1).run()
    assert out.status == "fallback"
    assert out.valid


def test_without_solver_there_is_no_referee():
    sc = scenario("05_multi_city_flights.json")
    out = Negotiation(sc, heuristic(sc), use_solver=False).run()
    assert not any(e.speaker == "Referee" for e in out.transcript)
    assert out.invalid_caught == 0
    if out.status == "agreed":
        assert out.verdict is not None  # still scored afterwards


class Scripted(Agent):
    """Agent that proposes fixed strings and always accepts."""

    def __init__(self, person, sc, starts):
        super().__init__(person, sc)
        self.starts = list(starts)

    def propose(self, board):
        start, venue = self.starts.pop(0)
        return Proposal(self.person.id, start, venue, f"Proposal {start}")

    def respond(self, board, proposal):
        return Reply(self.person.id, True, "ok")


def test_malformed_proposal_is_not_considered():
    sc = scenario("01_simple_team_sync.json")
    a, r, p = sc.participants()
    agents = [Scripted(a, sc, [("tomorrow at 10", None)]),
              Scripted(r, sc, [("2026-11-04T11:00", None)]),
              Scripted(p, sc, [])]
    out = Negotiation(sc, agents).run()
    assert any("could not read the start time" in e.text for e in out.transcript)
    assert out.status == "agreed" and out.start == sc.parse_local("2026-11-04T11:00", sc.display_tz)


def test_llm_only_mode_can_agree_on_invalid_slot():
    # Agents that accept anything: without the referee an impossible slot gets through.
    sc = scenario("05_multi_city_flights.json")
    a, r, p = sc.participants()
    agents = [Scripted(a, sc, [("2026-11-03T10:00", "bom_office")]), Scripted(r, sc, []), Scripted(p, sc, [])]
    out = Negotiation(sc, agents, use_solver=False).run()
    assert out.status == "agreed"
    assert not out.valid

    agents = [Scripted(a, sc, [("2026-11-03T10:00", "bom_office")]),
              Scripted(r, sc, [("2026-11-03T14:00", "bom_office")]), Scripted(p, sc, [])]
    out = Negotiation(sc, agents, use_solver=True).run()
    assert out.status == "agreed" and out.valid
    assert out.invalid_caught == 1


# --- LLM agent with a fake Claude client ----------------------------------------

class FakeMessages:
    def __init__(self, outputs, stop_reason="end_turn"):
        self.outputs = list(outputs)
        self.calls = []
        self.stop_reason = stop_reason

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = json.dumps(self.outputs.pop(0)) if self.outputs else "{}"
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=50),
        )


def fake_llm(outputs, stop_reason="end_turn"):
    messages = FakeMessages(outputs, stop_reason)
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return ClaudeLLM(client=client), messages


def test_llm_agents_negotiate_with_referee():
    sc = scenario("05_multi_city_flights.json")
    llm, fake = fake_llm([
        {"message": "Mumbai Tuesday morning?", "start": "2026-11-03T10:00", "venue": "bom_office"},  # Asha: invalid
        {"message": "Then Tuesday 14:00 here in Mumbai.", "start": "2026-11-03T14:00", "venue": "bom_office"},  # Rahul
        {"decision": "accept", "message": "Works for me.", "counter_start": "", "counter_venue": ""},  # Asha
        {"decision": "accept", "message": "Fine.", "counter_start": "", "counter_venue": ""},  # Priya
    ])
    agents = [LLMAgent(p, sc, llm) for p in sc.participants()]
    out = Negotiation(sc, agents).run()

    assert out.status == "agreed" and out.valid
    assert out.invalid_caught == 1
    assert (out.start, out.venue) == (sc.parse_local("2026-11-03T14:00", sc.display_tz), "bom_office")
    assert llm.usage.calls == 4

    # The referee's reasons reached the next agent's prompt.
    assert "cannot reach Mumbai office" in fake.calls[1]["messages"][0]["content"]


def test_llm_request_shape():
    sc = scenario("05_multi_city_flights.json")
    llm, fake = fake_llm([{"message": "x", "start": "2026-11-03T14:00", "venue": "bom_office"}])
    LLMAgent(sc.people["asha"], sc, llm).propose(Negotiation(sc, []).board)
    call = fake.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["fallbacks"] == "default"
    fmt = call["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["properties"]["venue"]["enum"] == sc.meeting.venues


def test_llm_agent_profile_is_private():
    sc = scenario("05_multi_city_flights.json")
    llm, _ = fake_llm([])
    asha = LLMAgent(sc.people["asha"], sc, llm)
    assert "Board prep" in asha.system          # her own event
    assert "Client lunch" not in asha.system    # Rahul's event
    assert "about 305 min (flight needed)" in asha.system


def test_llm_refusal_stops_negotiation():
    sc = scenario("01_simple_team_sync.json")
    llm, _ = fake_llm([{}], stop_reason="refusal")
    agents = [LLMAgent(p, sc, llm) for p in sc.participants()]
    out = Negotiation(sc, agents).run()
    assert out.status == "error"
    assert "declined" in out.error
