# Geo-Aware Multi-Agent Meeting Scheduler

AID 851 (Machine Reasoning) project. LLM agents negotiate a meeting time, and a Z3 solver checks that every proposal is valid. See [PLAN.md](PLAN.md) for the roadmap.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

## Run
```bash
python -m scheduler.cli data/scenarios/01_simple_team_sync.json
python -m scheduler.cli data/scenarios/01_simple_team_sync.json --verify 2026-11-03T10:00 -k 5
python -m scheduler.cli data/scenarios/05_multi_city_flights.json --verify 2026-11-03T10:00@bom_office
python -m pytest
```

### Negotiation (Phase 3)
```bash
# Offline: rule-based agents, no API key needed
python -m scheduler.negotiate data/scenarios/05_multi_city_flights.json --agents heuristic

# Claude agents (needs ANTHROPIC_API_KEY)
set ANTHROPIC_API_KEY=sk-ant-...        # Windows cmd   (PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-...")
python -m scheduler.negotiate data/scenarios/05_multi_city_flights.json
python -m scheduler.negotiate data/scenarios/05_multi_city_flights.json --no-solver   # LLM-only baseline
```
Options:
- `--max-rounds N` (default 8)
- `--model` (default `claude-opus-5-5`)
- `--effort low|medium|high|xhigh|max` (default `low`, to keep cost down)
- `--save PATH`

Each run prints the conversation live and writes a JSON log to `runs/`.

## Layout
| Path | Purpose |
|---|---|
| `scheduler/models.py` | Scenario data model and JSON loader. All times become integer minutes since the horizon start (UTC) |
| `scheduler/geo.py` | Locations, ground-travel estimates, flight timetable search |
| `scheduler/solver.py` | Z3 encoding: `verify`, `suggest`, `diagnose`, `explain` |
| `scheduler/cli.py` | Command-line runner for the solver |
| `scheduler/protocol.py` | Shared public board: proposals, replies, referee messages |
| `scheduler/agents.py` | `LLMAgent` (Claude) and `HeuristicAgent` (offline), one per participant |
| `scheduler/llm.py` | Claude API wrapper: structured JSON output, refusal fallbacks, token usage |
| `scheduler/negotiation.py` | Negotiation loop with the solver as referee (or without, as a baseline) |
| `scheduler/negotiate.py` | Command-line runner for negotiations |
| `data/scenarios/` | Scenarios 01–03 (time only) and 04–06 (geography) |
| `data/geo/world.json` | Offices, airports, travel parameters and an illustrative flight timetable |
| `tests/` | Unit tests, including a brute-force cross-check of the solver |

## How the solver works
- **Variable:** `S` is the meeting start (minutes). The meeting occupies `[S, S + duration)`.
- **Hard constraints** are labelled Z3 formulas over `S`:
  - within the planning horizon
  - on the slot grid
  - inside each person's working hours
  - no overlap with calendar events
  - each person's own hard rules
- **Soft preferences** are weighted formulas, maximised with `z3.Optimize` (weighted MaxSMT). Ties go to the earliest slot.
- **`verify(start)`** evaluates every labelled formula at `S = start` and lists every rule that is broken.
- **`suggest(k)`** returns the k best valid slots, spaced at least one meeting-length apart.
- **`diagnose()`** handles the case where no slot exists: it returns a minimal unsat core, i.e. the smallest set of rules that clash.
- **`explain(...)`** turns verdicts into plain English. These explanations are what the LLM agents will receive in Phase 3.

### Geography (Phase 2)
- **Second variable:** `V` is the index of the chosen venue in `meeting.venues`. A venue can be an office or `online`.
- **Anchors:** each person's day is a chain of fixed points:
  1. leave base (not before `earliest_departure`, default 06:00)
  2. calendar events, each at its `location`
  3. be back at base by `return_by` (default 23:00)

  The meeting must fit in a gap between two anchors.
- **Precomputed travel times:** for every (gap, venue), `geo.py` works out two numbers in plain Python, which enter Z3 as constants:
  - **EA**, the earliest arrival at the venue, by road or using the flight timetable;
  - **LD**, the latest departure from the venue that still makes the next anchor.

  Each person then gets two labelled constraints:
  - `reach_venue`: `in_gap ∧ V=v → S ≥ EA`
  - `onward_travel`: `in_gap ∧ V=v → S+D ≤ LD`
- **Objective order:** preference score, then total travel (`travel_objective: "total"`; use `"max"` for fairness or `"none"` to switch it off), then the earliest start.
- **Valid proposals** come with a travel plan: the latest possible departure, the exact flights, and the trip onward or home.
- **Simplifications:**
  - Ground time is great-circle distance × road factor, at city/highway speeds.
  - Trips over `max_drive_km` must fly, and only direct flights are used.
  - The timetable repeats daily.
  - Online participants can join from wherever they are.

### Negotiation (Phase 3)
- **One agent per participant.** An agent sees only its own person's calendar, rules, preferences and travel times, plus a shared public board.
- **Each round:**
  1. An agent proposes a time (and venue).
  2. The **referee** (the full solver) verifies it. If it is invalid, the solver's plain-language reasons are posted to the board and the proposal never reaches the group.
  3. Otherwise every other agent accepts or rejects, optionally with a counter-proposal.
  4. If everyone accepts, the meeting is agreed.
- **Getting unstuck:** after 4 rounds without agreement the referee posts 3 valid slots as hints. If the round limit is reached, it falls back to the solver's best slot. Impossible scenarios are diagnosed before any agent is called.
- **Claude agents:**
  - Each turn is one `claude-opus-5-5` call with structured JSON output (`output_config.format`). The schema limits the venue to the candidate ids.
  - Server-side refusal fallbacks are on (`fallbacks: "default"`), and the system prompt is cached.
  - Privacy: agents say "I'm busy then", and the referee's public messages leave out preferences and travel plans.
- **`--no-solver`** removes the referee. The agreed slot is still checked afterwards, so the LLM-only baseline can be scored in Phase 4.
- **Offline agents:** `HeuristicAgent` is a rule-based stand-in that uses a one-person solver for its own constraints. It makes runs reproducible and free, and powers the tests.

## Scenario format
```jsonc
{
  "name": "...", "description": "...",
  "timezone": "Asia/Kolkata",                 // display / horizon time zone
  "horizon": { "start": "2026-11-02", "days": 5 },
  "slot_minutes": 15,
  "meeting": { "title": "...", "duration_minutes": 60, "participants": ["asha", "rahul"] },
  "people": [{
    "id": "asha", "name": "Asha", "timezone": "Asia/Kolkata",
    "working_hours": { "start": "09:00", "end": "18:00", "days": ["Mon","Tue","Wed","Thu","Fri"] },
    "calendar": [{ "title": "Client call", "start": "2026-11-04T10:00", "end": "2026-11-04T11:00" }],
    "hard_constraints": [{ "type": "no_meetings_on_days", "days": ["Mon"] }],
    "preferences":      [{ "type": "prefer_time_window", "start": "09:00", "end": "12:00", "weight": 3 }]
  }]
}
```
Calendar times are given in the person's own time zone.

Geography fields (all optional; if `venues` is missing, the scenario behaves as in Phase 1):
```jsonc
{
  "geo": "../geo/world.json",                       // relative to the scenario file
  "meeting": { "venues": ["bom_office", "del_office", "online"], "travel_objective": "total" },
  "people": [{
    "base": "blr_ec_office",                        // required when venues are given
    "earliest_departure": "06:00", "return_by": "23:00",   // return_by: null = no same-day return
    "calendar": [{ "title": "...", "start": "...", "end": "...", "location": "blr_kor_office" }]
  }]
}
```

### Rule types
Each rule type below can be used as a hard constraint or as a soft preference (soft preferences also take a `weight`).

| Hard type | Soft counterpart | Parameters | Meaning |
|---|---|---|---|
| `only_between` | `prefer_time_window` | `start`, `end` | The meeting must lie inside this daily window |
| `blocked_window` | `avoid_window` | `start`, `end`, optional `days`, `label` | No overlap with this daily window (e.g. lunch) |
| `no_meetings_on_days` | `avoid_days` | `days` | No overlap with these weekdays |
| — | `prefer_days` | `days` | The meeting falls on one of these weekdays |
| `unavailable` | — | `start`, `end` (datetimes), `label` | Absent for a date range (e.g. leave) |
| — | `buffer_around_events` | `minutes` | Leave a gap around existing events |
| `in_person_only` | `prefer_in_person` | — | The venue is not `online` |
| `no_flying` | `avoid_flying` | — | The venue can be reached from base without flying |
| `max_travel_minutes` | `prefer_max_travel_minutes` | `minutes` | One-way door-to-door travel from base is within the limit |
| `allowed_venues` | `prefer_venues` | `venues` | The venue is one of these |
