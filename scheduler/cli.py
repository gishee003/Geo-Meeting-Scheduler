"""Command-line runner for the solver.

Examples
    python -m scheduler.cli data/scenarios/01_simple_team_sync.json
    python -m scheduler.cli data/scenarios/01_simple_team_sync.json --verify 2026-11-03T10:00
    python -m scheduler.cli data/scenarios/05_multi_city_flights.json --verify 2026-11-03T10:00@bom_office
"""

from __future__ import annotations

import argparse
import sys

from .models import load_scenario
from .solver import MeetingSolver


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Solve / verify a meeting-scheduling scenario.")
    ap.add_argument("scenario", help="path to scenario JSON")
    ap.add_argument("-k", type=int, default=3, help="number of suggestions (default 3)")
    ap.add_argument("--verify", metavar="YYYY-MM-DDTHH:MM[@venue]", action="append", default=[],
                    help="verify a proposed start time (scenario time zone) and venue; repeatable")
    args = ap.parse_args(argv)

    sc = load_scenario(args.scenario)
    solver = MeetingSolver(sc)
    names = ", ".join(p.name for p in sc.participants())

    print(f"=== {sc.name} ===")
    print(sc.description)
    print(f"Meeting: {sc.meeting.title}, {sc.meeting.duration} min, with {names}")
    if sc.geo_enabled:
        print(f"Venues: {', '.join(sc.location_name(v) for v in sc.meeting.venues)}")
    print(f"Rules: {len(solver.hard)} hard constraints, {len(solver.soft)} soft preferences\n")

    for text in args.verify:
        when, _, venue = text.partition("@")
        print(f"--- Verify proposal {text} ---")
        verdict = solver.verify(sc.parse_local(when, sc.display_tz), venue or None)
        print(solver.explain(verdict), "\n")

    diag = solver.diagnose()
    if not diag.feasible:
        print(solver.explain_diagnosis(diag))
        return 1

    for i, opt in enumerate(solver.suggest(args.k), 1):
        print(f"--- Suggestion {i} ---")
        print(solver.explain(opt.verdict), "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
