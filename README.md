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
python -m pytest
```

## Layout
| Path | Purpose |
|---|---|
| `scheduler/models.py` | Scenario data model and JSON loader. All times become integer minutes since the horizon start (UTC) |
| `scheduler/solver.py` | Z3 encoding: `verify`, `suggest`, `diagnose`, `explain` |
| `scheduler/cli.py` | Command-line runner |
| `data/scenarios/` | Test scenarios (easy, trade-off, impossible) |
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
