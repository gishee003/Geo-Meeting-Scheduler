"""Logic behind the demo app (app.py), kept free of Streamlit so it can be tested.

The app can either run a negotiation live (Gemini or Ollama agents) or replay
a saved Phase 4 run from results/logs, which needs no internet or API key.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .agents import SYSTEM_KINDS, HeuristicAgent, LLMAgent, personal_solver
from .evaluate import TIERS, run_solver_only
from .llm import DEFAULT_GEMINI_LITE_MODEL, DEFAULT_OLLAMA_MODEL, FallbackLLM, GeminiLLM, OllamaLLM
from .models import DAY_NAMES, Scenario, load_scenario, with_truth
from .negotiation import Negotiation, Outcome
from .protocol import Entry
from .solver import MeetingSolver, Option, Verdict
from .translate import translate_scenario

ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = ROOT / "data" / "scenarios"
RESULTS_DIR = ROOT / "results"

MODES = {
    "hybrid": "Hybrid: AI agents + solver referee (ours)",
    "llm_only": "LLM only: AI agents, no referee",
    "heuristic": "Rule-based agents + solver referee",
    "solver_only": "Solver only: no negotiation",
}
LLM_MODES = ("hybrid", "llm_only")
PROVIDERS = {
    f"Gemini Flash-Lite ({DEFAULT_GEMINI_LITE_MODEL})": f"gemini:{DEFAULT_GEMINI_LITE_MODEL}",
    f"Local Ollama ({DEFAULT_OLLAMA_MODEL})": f"ollama:{DEFAULT_OLLAMA_MODEL}",
}

# Chart colors (validated palette; status colors carry a text label too).
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BUSY, WORK, MEETING_OK, MEETING_BAD = "#a3a29c", "#e4eefb", "#2a78d6", "#d03b3b"


# --- scenarios ---------------------------------------------------------------------

def list_scenarios() -> list[tuple[str, str, str]]:
    """(file stem, display name, tier) for every scenario."""
    out = []
    for path in sorted(SCENARIO_DIR.glob("*.json")):
        sc = load_scenario(path)
        out.append((path.stem, sc.name, TIERS.get(path.stem, "")))
    return out


def load(stem: str) -> Scenario:
    return load_scenario(SCENARIO_DIR / f"{stem}.json")


def people_rows(sc: Scenario) -> list[dict]:
    """One readable row per participant for the scenario overview."""
    rows = []
    for p in sc.participants():
        me = personal_solver(sc, p.id)
        rules = [c.text for c in me.hard
                 if c.kind not in SYSTEM_KINDS | {"working_hours", "busy", "reach_venue", "onward_travel"}]
        wh = p.working_hours
        rows.append({
            "Person": p.name,
            "Based at": sc.location_name(p.base) if p.base else "-",
            "Time zone": p.tz.key,
            "Working hours": f"{', '.join(wh.days)} {wh.start:%H:%M}-{wh.end:%H:%M}",
            "Calendar": "; ".join(f"{sc.fmt(e.start, p.tz)} {e.title}" for e in p.calendar) or "-",
            "Must-follow rules": "; ".join(r.split(f"{p.name} ", 1)[-1] for r in rules) or "-",
            "Preferences": "; ".join(f"{c.text.split(f'{p.name} ', 1)[-1]} (w={c.weight})" for c in me.soft) or "-",
            "Notes (plain English)": " ".join(f'"{n}"' for n in p.notes) or "-",
        })
    return rows


# --- checking results ---------------------------------------------------------------

@dataclass
class Check:
    verdict: Verdict | None
    explanation: str
    best: Option | None
    max_score: int


def final_check(sc: Scenario, start: int | None, venue: str | None) -> Check:
    """Verify a final slot against everything people really need, and compare with the best slot."""
    truth = MeetingSolver(with_truth(sc))
    best = truth.suggest(1)
    max_score = sum(c.weight for c in truth.soft)
    if start is None:
        return Check(None, "", best[0] if best else None, max_score)
    verdict = truth.verify(start, venue)
    return Check(verdict, truth.explain(verdict), best[0] if best else None, max_score)


def slot_text(sc: Scenario, start: int, venue: str | None) -> str:
    end = sc.to_datetime(start + sc.meeting.duration)
    text = f"{sc.fmt(start)}-{end:%H:%M} ({sc.display_tz.key})"
    return text + (f" at {sc.location_name(venue)}" if venue else "")


# --- saved runs ------------------------------------------------------------------------

def saved_runs(stem: str, system: str) -> list[Path]:
    return sorted((RESULTS_DIR / "logs").glob(f"{stem}__{system}__r*.json"))


def load_saved(path: Path) -> tuple[dict, list[Entry]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["row"], [Entry(**e) for e in data["transcript"]]


def phase4_rows(stem: str) -> list[dict]:
    path = RESULTS_DIR / "runs.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["scenario"] == stem]


def phase4_summary(stem: str) -> list[dict]:
    """Per-system results of the Phase 4 experiment for one scenario."""
    rows = phase4_rows(stem)
    out = []
    for system, label in MODES.items():
        rs = [r for r in rows if r["system"] == system]
        if not rs:
            continue
        calls = [int(r["llm_calls"]) for r in rs]
        out.append({
            "System": label,
            "Runs": len(rs),
            "Correct": f"{sum(r['correct'] == 'True' for r in rs)}/{len(rs)}",
            "Impossible meeting scheduled": sum(r["unsafe"] == "True" for r in rs),
            "Avg rounds": round(sum(int(r["rounds"]) for r in rs) / len(rs), 1),
            "Avg AI calls": round(sum(calls) / len(rs), 1),
        })
    return out


# --- live runs -------------------------------------------------------------------------

def make_llm(specs: list[str]) -> FallbackLLM:
    """Provider chain from 'gemini:<model>' / 'ollama:<model>' entries; Gemini needs GEMINI_API_KEY."""
    chain = []
    for spec in specs:
        name, _, model = spec.partition(":")
        if name == "gemini":
            if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
                chain.append(GeminiLLM(model=model))
        elif name == "ollama":
            chain.append(OllamaLLM(model=model))
    if not chain:
        raise ValueError("No AI provider available: add GEMINI_API_KEY to .env or choose Ollama")
    return FallbackLLM(chain)


def run_live(sc: Scenario, mode: str, specs: list[str], max_rounds: int,
             on_post: Callable[[Entry], None]) -> tuple[Outcome, FallbackLLM | None]:
    """Run one negotiation, streaming each message to `on_post`."""
    if mode == "solver_only":
        return run_solver_only(MeetingSolver(sc)), None
    if mode == "heuristic":
        agents = [HeuristicAgent(p, sc) for p in sc.participants()]
        return Negotiation(sc, agents, max_rounds=max_rounds, on_post=on_post).run(), None
    llm = make_llm(specs)
    intro = []
    if mode == "hybrid":
        sc, records = translate_scenario(sc, llm)
        intro = [r.summary() for r in records]
    agents = [LLMAgent(p, sc, llm, mode == "hybrid") for p in sc.participants()]
    neg = Negotiation(sc, agents, use_solver=mode == "hybrid", max_rounds=max_rounds, on_post=on_post, intro=intro)
    return neg.run(), llm


# --- charts ------------------------------------------------------------------------------

def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=8)


def calendar_figure(sc: Scenario, start: int | None = None, venue: str | None = None, valid: bool = True,
                    first_hour: int = 6, last_hour: int = 23):
    """Week view in the scenario's time zone: working hours, busy events and the meeting."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    people = sc.participants()
    days = sc.horizon_days
    fig, axes = plt.subplots(1, days, figsize=(2.3 * days + 1.2, 0.55 * len(people) + 1.4),
                             sharey=True, facecolor=SURFACE, squeeze=False)
    for i, ax in enumerate(axes[0]):
        midnight = i * 24 * 60
        lo, hi = midnight + first_hour * 60, midnight + last_hour * 60

        def bar(y, a, b, color, height):
            a, b = max(a, lo), min(b, hi)
            if b > a:
                ax.barh(y, (b - a) / 60, left=(a - midnight) / 60, height=height, color=color,
                        edgecolor=SURFACE, linewidth=1.5)

        for y, p in enumerate(people):
            for d, _ in sc.local_days(p.tz):
                if DAY_NAMES[d.weekday()] in p.working_hours.days:
                    a, b = sc.local_window(d, p.working_hours.start, p.working_hours.end, p.tz)
                    bar(y, a, b, WORK, 0.8)
            for ev in p.calendar:
                bar(y, ev.start, ev.end, BUSY, 0.5)
        if start is not None:
            a, b = max(start, lo), min(start + sc.meeting.duration, hi)
            if b > a:
                ax.axvspan((a - midnight) / 60, (b - midnight) / 60, color=MEETING_OK if valid else MEETING_BAD,
                           alpha=0.85, zorder=3)
        ax.set_xlim(first_hour, last_hour)
        ax.set_xticks(range(first_hour, last_hour + 1, 3), [f"{h:02d}" for h in range(first_hour, last_hour + 1, 3)])
        ax.set_title(sc.to_datetime(midnight).strftime("%a %d %b"), color=INK, fontsize=9)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        _style(ax)
    axes[0][0].set_yticks(range(len(people)), [p.name for p in people], color=INK, fontsize=9)
    axes[0][0].invert_yaxis()
    legend = [Patch(color=WORK, label="Working hours"), Patch(color=BUSY, label="Busy")]
    if start is not None:
        legend.append(Patch(color=MEETING_OK if valid else MEETING_BAD,
                            label="Meeting (valid)" if valid else "Meeting (INVALID)"))
    fig.legend(handles=legend, loc="lower center", ncol=len(legend), frameon=False, fontsize=8, labelcolor=MUTED)
    fig.text(0.01, 0.98, f"Hours in {sc.display_tz.key}", color=MUTED, fontsize=8, va="top")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    return fig


def map_figure(sc: Scenario, venue: str | None = None):
    """Where everyone is based and, if in person, where they travel to. None without geography."""
    if not sc.geo_enabled:
        return None
    import matplotlib.pyplot as plt

    loc = sc.travel.loc
    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor=SURFACE)
    target = loc[venue] if venue and not loc[venue].online else None
    bases = {}
    for p in sc.participants():
        bases.setdefault(p.base, []).append(p.name)
    for base_id, names in bases.items():
        b = loc[base_id]
        if target and base_id != venue:
            ax.plot([b.lon, target.lon], [b.lat, target.lat], color=MEETING_OK, linewidth=1.5, alpha=0.6, zorder=1)
        ax.scatter(b.lon, b.lat, s=70, color=INK, zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.annotate(f"{', '.join(names)}\n{b.city}", (b.lon, b.lat), textcoords="offset points", xytext=(8, 6),
                    fontsize=8, color=INK)
    if target:
        ax.scatter(target.lon, target.lat, s=260, marker="*", color=MEETING_OK, zorder=4, edgecolor=SURFACE,
                   linewidth=1.5)
        ax.annotate(f"Venue: {target.name}", (target.lon, target.lat), textcoords="offset points", xytext=(8, -14),
                    fontsize=8, color=MEETING_OK, fontweight="bold")
    elif venue:
        ax.set_title("Online meeting: nobody travels", color=MUTED, fontsize=9, loc="left")
    ax.margins(0.25)
    ax.set_xlabel("Longitude", color=MUTED, fontsize=8)
    ax.set_ylabel("Latitude", color=MUTED, fontsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    _style(ax)
    fig.tight_layout()
    return fig
