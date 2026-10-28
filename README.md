# Geo-Aware Multi-Agent Meeting Scheduler

AID 851 (Machine Reasoning) project. LLM agents negotiate a meeting time, and a Z3 solver checks that every proposal is valid. See [PLAN.md](PLAN.md) for the roadmap.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
copy .env.example .env          # then put your GEMINI_API_KEY in .env
ollama pull qwen3:4b            # optional: local fallback model (needs Ollama installed)
```

## Quick test commands (Windows PowerShell)
Run these from the project folder.

```powershell
# 1. Normal run: Gemini agents with the referee, falling back to Ollama if needed
.venv\Scripts\python -m scheduler.negotiate data\scenarios\05_multi_city_flights.json

# 2. Without the referee: the "AI alone" baseline
.venv\Scripts\python -m scheduler.negotiate data\scenarios\05_multi_city_flights.json --no-solver

# 3. Only the local model (run 'ollama pull qwen3:4b' first; close heavy apps)
.venv\Scripts\python -m scheduler.negotiate data\scenarios\04_same_city_travel.json --providers ollama

# 4. Offline rule-based agents: no AI, instant
.venv\Scripts\python -m scheduler.negotiate data\scenarios\05_multi_city_flights.json --agents heuristic

# 5. All automatic tests
.venv\Scripts\python -m pytest -q
```

Swap the scenario file in any command to try another case:

| File | What it tests |
|---|---|
| `01_simple_team_sync.json` | One office, calendars only |
| `02_busy_week.json` | Packed calendars, trade-offs between preferences |
| `03_impossible.json` | No possible slot; should stop immediately |
| `04_same_city_travel.json` | Two offices 50 min apart |
| `05_multi_city_flights.json` | Three cities, flights there and back |
| `06_london_time_zones.json` | London plus India; should end up online |

Compare commands 1 and 2: without the referee, check whether the agents agree on a slot that is actually impossible. The final summary marks it **INVALID**. Every run is saved as a JSON log in `runs\`.

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

# LLM agents, using the first provider that works: Gemini Flash -> Gemini Flash-Lite -> Ollama
python -m scheduler.negotiate data/scenarios/05_multi_city_flights.json
python -m scheduler.negotiate data/scenarios/05_multi_city_flights.json --providers ollama
python -m scheduler.negotiate data/scenarios/05_multi_city_flights.json --no-solver   # LLM-only baseline
```

**LLM providers.** Copy `.env.example` to `.env` (git-ignored) and add your Gemini key. A provider without a key is skipped. If a provider fails mid-run (quota, network), the run switches to the next one and stays there.

| Provider | Needs | Default model | Flag |
|---|---|---|---|
| Gemini | `GEMINI_API_KEY` (free tier at aistudio.google.com) | `gemini-3.8-flash`, then `gemini-3.5-flash-lite` | `--gemini-model` |
| Ollama | [Ollama](https://ollama.com) installed, then `ollama pull qwen3:4b` | `qwen3:4b` (needs about 3 GB of free RAM) | `--ollama-model`, `--ollama-think` |

Temporary errors (server busy, rate limit, network) are retried twice, after 5 s and 15 s, before the run moves on to the next provider.

Other options:
- `--providers` sets the fallback order as comma-separated `name` or `name:model` entries (default `gemini,gemini:gemini-3.5-flash-lite,ollama`)
- `--max-rounds N` (default 8)
- `--save PATH`

Each run prints the conversation live and writes a JSON log to `runs/`. The log records which provider answered each run and any provider failures.

## Layout
| Path | Purpose |
|---|---|
| `scheduler/models.py` | Scenario data model and JSON loader. All times become integer minutes since the horizon start (UTC) |
| `scheduler/geo.py` | Locations, ground-travel estimates, flight timetable search |
| `scheduler/solver.py` | Z3 encoding: `verify`, `suggest`, `diagnose`, `explain` |
| `scheduler/cli.py` | Command-line runner for the solver |
| `scheduler/protocol.py` | Shared public board: proposals, replies, referee messages |
| `scheduler/agents.py` | `LLMAgent` (Gemini or Ollama) and `HeuristicAgent` (offline), one per participant |
| `scheduler/llm.py` | LLM providers (Gemini, Ollama) with JSON-schema output, plus the fallback chain |
| `scheduler/env.py` | Loads API keys from a git-ignored `.env` file |
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
- **LLM agents:**
  - Each turn is one LLM call constrained to a JSON schema; the schema limits the venue to the candidate ids. Gemini uses `response_json_schema` and Ollama uses `format`.
  - Temporary provider errors are retried, and each call has a time limit, so a busy model cannot hang a run.
  - Privacy: agents say "I'm busy then", and the referee's public messages leave out preferences and travel plans.
- **When each outcome is decided.** One round = one proposal checked, plus the replies if it is valid.

  | Outcome | With the referee (default) | Without the referee (`--no-solver`) |
  |---|---|---|
  | **Impossible** | Before round 1: no valid slot exists, so the run stops (0 rounds, no LLM calls) | Not detected; the agents keep trying |
  | **Invalid proposal** | In the round it is made: rejected with reasons, and the next agent proposes | Not checked; it can be accepted |
  | **Valid proposal** | In the round it is made: the other agents vote on it | Not checked |
  | **Agreed** | Any round from 1 to `--max-rounds` (default 8), when everyone accepts; always valid | Any round; validity is checked once afterwards and reported as VALID or INVALID |
  | **Hint** | After round 4 without agreement: 3 valid slots are posted | — |
  | **Fallback** | After the last round: the solver's best valid slot is used | Status `no_agreement` (no fallback) |
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
