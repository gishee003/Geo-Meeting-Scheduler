"""Run a negotiation between agents, refereed by the solver.

Examples
    python -m scheduler.negotiate data/scenarios/05_multi_city_flights.json --agents heuristic
    python -m scheduler.negotiate data/scenarios/05_multi_city_flights.json             # LLM agents
    python -m scheduler.negotiate data/scenarios/05_multi_city_flights.json --providers gemini,ollama
    python -m scheduler.negotiate data/scenarios/05_multi_city_flights.json --no-solver  # LLM-only baseline

API keys are read from the environment or from a git-ignored .env file
(see .env.example). Providers without a key are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .agents import HeuristicAgent, LLMAgent
from .env import load_dotenv
from .llm import (DEFAULT_GEMINI_LITE_MODEL, DEFAULT_GEMINI_MODEL, DEFAULT_OLLAMA_MODEL, FallbackLLM,
                  GeminiLLM, OllamaLLM)
from .models import load_scenario
from .negotiation import Negotiation, Outcome

PROVIDERS = ("gemini", "ollama")
DEFAULT_CHAIN = f"gemini,gemini:{DEFAULT_GEMINI_LITE_MODEL},ollama"


def build_llm(args) -> FallbackLLM:
    """The provider chain from --providers ("name" or "name:model"), skipping cloud providers without a key."""
    chain = []
    for spec in args.providers.split(","):
        name, _, model = spec.strip().partition(":")
        name = name.lower()
        if name not in PROVIDERS:
            raise SystemExit(f"Unknown provider '{name}' (choose from {', '.join(PROVIDERS)})")
        if name == "gemini":
            if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
                print("[llm] skipping Gemini: GEMINI_API_KEY is not set")
                continue
            chain.append(GeminiLLM(model=model or args.gemini_model))
        else:
            chain.append(OllamaLLM(model=model or args.ollama_model, think=True if args.ollama_think else None))
    if not chain:
        raise SystemExit("No LLM provider available - set an API key, start Ollama, or use --agents heuristic")

    def switched(failed: str, reason: str, nxt: str | None) -> None:
        print(f"[llm] {failed} failed: {reason}")
        print(f"[llm] switching to {nxt}\n" if nxt else "[llm] no providers left\n")

    def retrying(name: str, reason: str, delay: float) -> None:
        print(f"[llm] {name}: {reason} - retrying in {delay:.0f}s")

    print(f"[llm] provider order: {' -> '.join(p.name for p in chain)}\n")
    return FallbackLLM(chain, on_switch=switched, on_retry=retrying)


def build_agents(sc, args, use_solver: bool):
    if args.agents == "heuristic":
        return [HeuristicAgent(p, sc) for p in sc.participants()], None
    llm = build_llm(args)
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
        "providers": [p.name for p in llm.providers] if llm else None,
        "served_by": {p.name: asdict(p.usage) for p in llm.providers if p.usage.calls} if llm else None,
        "provider_failures": llm.failures if llm else None,
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
    ap.add_argument("--agents", choices=["llm", "heuristic"], default="llm")
    ap.add_argument("--providers", default=DEFAULT_CHAIN,
                    help=f"LLM fallback order, comma-separated 'name' or 'name:model' (default: {DEFAULT_CHAIN})")
    ap.add_argument("--no-solver", action="store_true", help="LLM-only baseline: no referee during negotiation")
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    ap.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    ap.add_argument("--ollama-think", action="store_true", help="let the local model think first (slower)")
    ap.add_argument("--save", metavar="PATH", help="where to write the JSON log (default: runs/...)")
    args = ap.parse_args(argv)

    load_dotenv()
    sc = load_scenario(args.scenario)
    use_solver = not args.no_solver
    agents, llm = build_agents(sc, args, use_solver)

    def show(entry):
        print(f"{entry.speaker}: {entry.text}\n")

    print(f"=== {sc.name} | agents: {args.agents} | referee: {'on' if use_solver else 'off'} ===\n")
    neg = Negotiation(sc, agents, use_solver=use_solver, max_rounds=args.max_rounds, on_post=show)
    out = neg.run()
    print("=" * 60)
    print(summary(sc, neg.solver, out))
    if llm:
        for p in llm.providers:
            if p.usage.calls:
                u = p.usage
                print(f"LLM usage ({p.name}): {u.calls} calls, {u.input_tokens} input / "
                      f"{u.output_tokens} output tokens")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mode = "hybrid" if use_solver else "llm-only"
    path = Path(args.save or f"runs/{Path(args.scenario).stem}_{args.agents}_{mode}_{stamp}.json")
    save(path, args.scenario, args, out, llm)
    print(f"Log saved to {path}")
    return 0 if out.status in ("agreed", "fallback") else 1


if __name__ == "__main__":
    sys.exit(main())
