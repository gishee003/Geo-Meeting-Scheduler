"""Neural-to-symbolic step: turn plain-language notes into solver rules.

A solver cannot read "I pick up my kids at 15:30 on Tuesdays". The LLM converts
such notes into formal rules, which are validated and then enforced by the
solver like any other rule. The solver guarantees these rules hold, but not that
the translation is right, so every translated rule is shown in the conversation
and malformed ones are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time

from .llm import LLM
from .models import DAY_NAMES, Person, Rule, Scenario, with_rules
from .solver import MeetingSolver

KINDS = {  # kind -> (hard rule type, soft rule type)
    "only_between": ("only_between", "prefer_time_window"),
    "blocked_window": ("blocked_window", "avoid_window"),
    "avoid_days": ("no_meetings_on_days", "avoid_days"),
    "prefer_days": (None, "prefer_days"),
}

SCHEMA = {
    "type": "object",
    "properties": {"rules": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "note": {"type": "string"},
            "kind": {"type": "string", "enum": list(KINDS)},
            "strength": {"type": "string", "enum": ["hard", "soft"]},
            "start": {"type": "string"},
            "end": {"type": "string"},
            "days": {"type": "array", "items": {"type": "string", "enum": DAY_NAMES}},
            "importance": {"type": "integer"},
        },
        "required": ["note", "kind", "strength", "start", "end", "days", "importance"],
        "additionalProperties": False,
    }}},
    "required": ["rules"],
    "additionalProperties": False,
}

SYSTEM = """You convert a person's plain-English scheduling notes into formal rules for a logic solver.

Rule kinds:
- only_between: the meeting must lie inside start-end every day
- blocked_window: the meeting must not overlap start-end, on the listed days (empty days = every day)
- avoid_days: the meeting must not fall on the listed days
- prefer_days: the meeting should fall on one of the listed days

strength: "hard" for things the person cannot do ("can't", "must", "impossible", a fixed commitment); \
"soft" for wishes ("prefer", "rather", "ideally").
importance: 1 to 3 for soft rules (3 = matters most); use 1 for hard rules.
start and end: 24-hour HH:MM local time, or empty strings when the kind has no times.
days: use Mon, Tue, Wed, Thu, Fri, Sat, Sun.
note: copy the part of the note this rule comes from.

One note may need several rules. Include a commitment's travel or buffer time only if the note states it."""


@dataclass
class Translation:
    person: str
    note: str
    rule: Rule | None    # None if the model's answer could not be used
    text: str

    def summary(self) -> str:
        if self.rule is None:
            return f"Could not use a rule from {self.person}'s note: {self.text}"
        kind = "Must-follow rule" if self.rule.weight == 0 else "Preference"
        return f"{kind} from {self.person}'s note \"{self.note}\": {self.text}."


def _to_rule(raw: dict) -> Rule:
    """Validate one model-written rule. Raises ValueError if it cannot be used."""
    kind, hard = raw["kind"], raw["strength"] == "hard"
    rule_type = KINDS[kind][0 if hard else 1] or KINDS[kind][1]
    if rule_type == "prefer_days":
        hard = False
    params: dict = {}
    if kind in ("only_between", "blocked_window"):
        start, end = time.fromisoformat(raw["start"]), time.fromisoformat(raw["end"])
        if start >= end:
            raise ValueError(f"start {raw['start']} is not before end {raw['end']}")
        params |= {"start": raw["start"], "end": raw["end"]}
    if kind in ("avoid_days", "prefer_days") and not raw["days"]:
        raise ValueError("no days given")
    if raw["days"] and kind != "only_between":
        params["days"] = raw["days"]
    return Rule(rule_type, params, 0 if hard else max(1, min(3, int(raw.get("importance") or 1))))


def translate_person(p: Person, sc: Scenario, llm: LLM) -> tuple[Scenario, list[Translation]]:
    prompt = f"Person: {p.name}\nNotes:\n" + "\n".join(f'- "{n}"' for n in p.notes)
    raw_rules = llm.json(SYSTEM, prompt, SCHEMA)["rules"]
    solver = MeetingSolver(sc)
    records, hard, soft = [], [], []
    for raw in raw_rules:
        try:
            rule = _to_rule(raw)
            text = solver._encode_rule(p, rule, "note").text
        except (KeyError, ValueError, TypeError) as e:
            records.append(Translation(p.name, raw.get("note", ""), None, str(e)))
            continue
        (soft if rule.weight else hard).append(rule)
        records.append(Translation(p.name, raw.get("note", ""), rule, text))
    return with_rules(sc, p.id, hard, soft), records


def translate_scenario(sc: Scenario, llm: LLM) -> tuple[Scenario, list[Translation]]:
    """Scenario with everyone's notes added as solver rules, plus a record of each translation."""
    records = []
    for p in sc.participants():
        if p.notes:
            sc, recs = translate_person(p, sc, llm)
            records += recs
    return sc, records
