"""Phase 4 experiments: compare the systems on every scenario.

Systems
    llm_only    - LLM agents, no referee (final slot is checked afterwards)
    solver_only - Z3 alone: no negotiation, no LLM
    hybrid      - LLM agents with the solver as referee (our approach)
    heuristic   - rule-based agents with the referee (deterministic comparison)

The LLM systems are repeated because LLM replies vary between runs. Each run
is saved as one JSON log in <out>/logs, and <out>/runs.csv is rebuilt from
those logs, so an interrupted experiment resumes where it stopped. A single
model is used throughout: if it fails, the experiment stops rather than mixing
models, and the failed run is retried on the next start.

    python -m scheduler.evaluate
    python -m scheduler.evaluate --repeats 5 --systems llm_only,hybrid
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .agents import HeuristicAgent, LLMAgent
from .env import load_dotenv
from .llm import DEFAULT_GEMINI_LITE_MODEL, LLM, FallbackLLM, GeminiLLM, LLMError
from .models import Scenario, load_scenario, with_truth
from .negotiation import Negotiation, Outcome
from .solver import MeetingSolver
from .translate import translate_scenario

SCENARIOS = Path("data/scenarios")
SYSTEMS = ("llm_only", "solver_only", "hybrid", "heuristic")
LLM_SYSTEMS = ("llm_only", "hybrid")
TIERS = {
    "01_simple_team_sync": "easy",
    "02_busy_week": "medium",
    "03_impossible": "impossible",
    "04_same_city_travel": "easy",
    "05_multi_city_flights": "medium",
    "06_london_time_zones": "medium",
    "07_four_cities_dense": "hard",
    "08_global_five_people": "hard",
    "09_plain_language_notes": "plain-language",
}
FIELDS = ["scenario", "tier", "system", "repeat", "status", "correct", "unsafe", "valid", "rescued",
          "rounds", "proposals", "invalid_caught", "llm_calls", "input_tokens", "output_tokens",
          "score", "best_score", "max_score", "travel", "best_travel", "runtime_s", "model", "start", "venue",
          "notes_translated"]


@dataclass
class Truth:
    """What the solver alone says about a scenario."""
    feasible: bool
    best_score: int | None
    best_travel: int | None
    max_score: int


def ground_truth(solver: MeetingSolver) -> Truth:
    best = solver.suggest(1)
    max_score = sum(c.weight for c in solver.soft)
    if not best:
        return Truth(False, None, None, max_score)
    v = best[0].verdict
    return Truth(True, v.score, sum(v.travel.values()), max_score)


def is_correct(status: str, valid: bool, feasible: bool) -> bool:
    """Feasible scenario: a valid meeting. Impossible one: the system must say so."""
    if feasible:
        return status in ("agreed", "fallback", "solved") and valid
    return status == "impossible"


def is_unsafe(status: str, valid: bool) -> bool:
    """The system confidently scheduled a meeting that cannot happen or ignores what someone needs."""
    return status in ("agreed", "solved", "fallback") and not valid


def make_row(scenario: str, system: str, repeat: int, truth: Truth, out: Outcome,
             runtime: float, usage=(0, 0, 0), model: str | None = None) -> dict:
    v = out.verdict
    return {
        "scenario": scenario,
        "tier": TIERS.get(scenario, "unknown"),
        "system": system,
        "repeat": repeat,
        "status": out.status,
        "correct": is_correct(out.status, out.valid, truth.feasible),
        "unsafe": is_unsafe(out.status, out.valid),
        "valid": out.valid,
        "rescued": out.status == "fallback",
        "rounds": out.rounds,
        "proposals": out.proposals,
        "invalid_caught": out.invalid_caught,
        "llm_calls": usage[0],
        "input_tokens": usage[1],
        "output_tokens": usage[2],
        "score": v.score if v else None,
        "best_score": truth.best_score,
        "max_score": truth.max_score,
        "travel": sum(v.travel.values()) if v else None,
        "best_travel": truth.best_travel,
        "runtime_s": round(runtime, 3),
        "model": model,
        "start": v.start if v else None,
        "venue": out.venue,
    }


def run_solver_only(solver: MeetingSolver) -> Outcome:
    best = solver.suggest(1)
    if not best:
        return Outcome("impossible")
    return Outcome("solved", best[0].start, best[0].venue, best[0].verdict)


def run_heuristic(sc: Scenario) -> Outcome:
    return Negotiation(sc, [HeuristicAgent(p, sc) for p in sc.participants()]).run()


def run_llm(sc: Scenario, llm: LLM, use_solver: bool, max_rounds: int = 8) -> tuple[Outcome, list]:
    """Hybrid first turns plain-language notes into solver rules; LLM-only agents just read them."""
    records, intro = [], []
    if use_solver:
        try:
            sc, records = translate_scenario(sc, llm)
        except LLMError as e:
            return Outcome("error", error=str(e)), []
        intro = [r.summary() for r in records]
    agents = [LLMAgent(p, sc, llm, use_solver) for p in sc.participants()]
    return Negotiation(sc, agents, use_solver=use_solver, max_rounds=max_rounds, intro=intro).run(), records


def run_once(name: str, sc: Scenario, system: str, repeat: int, truth: Truth,
             llm: LLM | None, model: str | None, max_rounds: int) -> tuple[dict, list]:
    records = []
    began = time.perf_counter()
    before = (llm.usage.calls, llm.usage.input_tokens, llm.usage.output_tokens) if llm else (0, 0, 0)
    if system == "solver_only":
        out = run_solver_only(MeetingSolver(sc))
    elif system == "heuristic":
        out = run_heuristic(sc)
    else:
        chain = FallbackLLM([llm])  # same model only; retries temporary errors, never switches model
        out, records = run_llm(sc, chain, system == "hybrid", max_rounds)
    runtime = time.perf_counter() - began
    if out.start is not None:   # score every system against what people really need
        out.verdict = MeetingSolver(with_truth(sc)).verify(out.start, out.venue)
    usage = (0, 0, 0)
    if llm:
        usage = (llm.usage.calls - before[0], llm.usage.input_tokens - before[1],
                 llm.usage.output_tokens - before[2])
    row = make_row(name, system, repeat, truth, out, runtime, usage, model if system in LLM_SYSTEMS else None)
    return row | {"error": out.error, "notes_translated": len(records)}, [asdict(e) for e in out.transcript]


def plan(names: list[str], systems: list[str], repeats: int) -> list[tuple[str, str, int]]:
    """Deterministic systems run once; LLM systems run `repeats` times."""
    jobs = []
    for repeat in range(1, repeats + 1):   # one full sweep per repeat, so partial results stay balanced
        for name in names:
            for system in systems:
                if repeat == 1 or system in LLM_SYSTEMS:
                    jobs.append((name, system, repeat))
    return jobs


def write_csv(out_dir: Path) -> int:
    rows = [json.loads(p.read_text(encoding="utf-8"))["row"] for p in sorted((out_dir / "logs").glob("*.json"))]
    with (out_dir / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the Phase 4 experiments.")
    ap.add_argument("--repeats", type=int, default=3, help="repeats of each LLM system (default 3)")
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--scenarios", help="comma-separated scenario names (default: all)")
    ap.add_argument("--model", default=DEFAULT_GEMINI_LITE_MODEL)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    load_dotenv()
    out_dir = Path(args.out)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    systems = args.systems.split(",")
    names = args.scenarios.split(",") if args.scenarios else sorted(p.stem for p in SCENARIOS.glob("*.json"))
    jobs = plan(names, systems, args.repeats)
    llm = GeminiLLM(model=args.model) if any(s in LLM_SYSTEMS for s in systems) else None

    print(f"{len(jobs)} runs planned, model {args.model}", flush=True)
    for i, (name, system, repeat) in enumerate(jobs, 1):
        log = out_dir / "logs" / f"{name}__{system}__r{repeat}.json"
        if log.exists():
            continue
        sc = load_scenario(SCENARIOS / f"{name}.json")
        truth = ground_truth(MeetingSolver(with_truth(sc)))
        row, transcript = run_once(name, sc, system, repeat, truth, llm, args.model, args.max_rounds)
        if row["status"] == "error":
            print(f"[{i}/{len(jobs)}] {name} {system} r{repeat}: STOPPED - {row['error']}", flush=True)
            write_csv(out_dir)
            print("Run the same command again later to resume.", flush=True)
            return 2
        log.write_text(json.dumps({"row": row, "transcript": transcript}, indent=2), encoding="utf-8")
        print(f"[{i}/{len(jobs)}] {name} {system} r{repeat}: {row['status']} valid={row['valid']} "
              f"rounds={row['rounds']} calls={row['llm_calls']}", flush=True)

    print(f"Done: {write_csv(out_dir)} runs in {out_dir / 'runs.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
