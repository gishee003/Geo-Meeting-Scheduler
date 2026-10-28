"""Run a negotiation between agents, refereed by the solver.

Examples
    python -m scheduler.negotiate data/scenarios/05_multi_city_flights.json --agents heuristic
    python -m scheduler.negotiate data/scenarios/05_multi_city_flights.json            # Claude agents
    python -m scheduler.negotiate data/scenarios/05_multi_city_flights.json --no-solver # LLM-only baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .agents import HeuristicAgent, LLMAgent
from .llm import DEFAULT_MODEL, ClaudeLLM
from .models import load_scenario
from .negotiation import Negotiation, Outcome


def build_agents(sc, kind: str, use_solver: bool, model: str, effort: str):
    if kind == "heuristic":
        return [HeuristicAgent(p, sc) for p in sc.participants()], None
    llm = ClaudeLLM(model=model, effort=effort)
    return [LLMAgent(p, sc, llm, use_solver) for p in sc.participants()], llm


def summary(sc, solver, out: Outcome) -> str:
    lines = [f"Outcome: {out.status.upper()} after {out.rounds} round(s), {out.proposals} proposal(s), "
             f"{out.invalid_caught} invalid proposal(s) caught by the referee"]
    if out.error:
        lines.append(f"Error: {out.error}")
    if out.verdict:
        lines.append(solver.explain(out.verdict))
    return "\n".join(lines)


def save(path: Path, scenario_path: str, args, out: Outcome, llm) -> None:
    record = {
        "scenario": scenario_path,
        "agents": args.agents,
        "use_solver": not args.no_solver,
        "model": args.model if args.agents == "claude" else None,
        "effort": args.effort if args.agents == "claude" else None,
        "status": out.status,
        "valid": out.valid,
        "start": out.verdict and out.verdict.start,
        "venue": out.venue,
        "score": out.verdict.score if out.verdict else None,
        "travel": out.verdict.travel if out.verdict else None,
        "rounds": out.rounds,
        "proposals": out.proposals,
        "invalid_caught": out.invalid_caught,
        "error": out.error,
        "usage": asdict(llm.usage) if llm else None,
        "transcript": [asdict(e) for e in out.transcript],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Negotiate a meeting with one agent per participant.")
    ap.add_argument("scenario")
    ap.add_argument("--agents", choices=["claude", "heuristic"], default="claude")
    ap.add_argument("--no-solver", action="store_true", help="LLM-only baseline: no referee during negotiation")
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--save", metavar="PATH", help="where to write the JSON log (default: runs/...)")
    args = ap.parse_args(argv)

    sc = load_scenario(args.scenario)
    use_solver = not args.no_solver
    agents, llm = build_agents(sc, args.agents, use_solver, args.model, args.effort)

    def show(entry):
        print(f"{entry.speaker}: {entry.text}\n")

    print(f"=== {sc.name} | agents: {args.agents} | referee: {'on' if use_solver else 'off'} ===\n")
    neg = Negotiation(sc, agents, use_solver=use_solver, max_rounds=args.max_rounds, on_post=show)
    out = neg.run()
    print("=" * 60)
    print(summary(sc, neg.solver, out))
    if llm:
        u = llm.usage
        print(f"LLM usage: {u.calls} calls, {u.input_tokens} input / {u.output_tokens} output tokens")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mode = "hybrid" if use_solver else "llm-only"
    path = Path(args.save or f"runs/{Path(args.scenario).stem}_{args.agents}_{mode}_{stamp}.json")
    save(path, args.scenario, args, out, llm)
    print(f"Log saved to {path}")
    return 0 if out.status in ("agreed", "fallback") else 1


if __name__ == "__main__":
    sys.exit(main())
