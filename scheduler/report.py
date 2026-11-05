"""Turn results/runs.csv into summary tables and charts.

    python -m scheduler.report

Writes results/summary.md and results/charts/*.png. The tables are the
accessible counterpart of the charts: every charted value appears in them.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

SYSTEM_LABELS = {"hybrid": "Hybrid (ours)", "llm_only": "LLM only", "solver_only": "Solver only",
                 "heuristic": "Rule-based agents"}
SYSTEM_ORDER = ["hybrid", "llm_only", "solver_only", "heuristic"]
TIER_ORDER = ["easy", "medium", "hard", "impossible", "plain-language"]

# Colors follow the system, never its rank (validated for color-blind separation).
SYSTEM_COLORS = {"hybrid": "#2a78d6", "llm_only": "#eb6834", "solver_only": "#1baf7a", "heuristic": "#eda100"}
OUTCOME_COLORS = {"correct": "#0ca30c", "unsafe": "#d03b3b", "none": "#a3a29c"}
OUTCOME_LABELS = {"correct": "Correct result", "unsafe": "Impossible meeting scheduled", "none": "No meeting, nothing proved"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in ("correct", "unsafe", "valid", "rescued"):
            r[key] = r[key] == "True"
        for key in ("repeat", "rounds", "proposals", "invalid_caught", "llm_calls", "input_tokens", "output_tokens",
                    "max_score"):
            r[key] = int(r[key])
        for key in ("score", "best_score", "travel", "best_travel"):
            r[key] = int(r[key]) if r[key] not in ("", None) else None
        r["runtime_s"] = float(r["runtime_s"])
    return rows


def outcome_of(r: dict) -> str:
    if r["correct"]:
        return "correct"
    return "unsafe" if r["unsafe"] else "none"


def score_ratio(r: dict) -> float | None:
    """Preference score relative to the solver's best, for valid meetings only."""
    if not r["valid"] or not r["best_score"]:
        return None
    return r["score"] / r["best_score"]


def avg(values) -> float | None:
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def fmt(value, digits: int = 1, suffix: str = "") -> str:
    return "–" if value is None else f"{value:.{digits}f}{suffix}"


def group(rows: list[dict], key: str) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return groups


def system_summary(rows: list[dict]) -> list[dict]:
    out = []
    groups = group(rows, "system")
    for system in SYSTEM_ORDER:
        rs = groups.get(system)
        if not rs:
            continue
        negotiated = system != "solver_only"
        ratios = [score_ratio(r) for r in rs]
        extra = [r["travel"] - r["best_travel"] for r in rs if r["valid"] and r["best_travel"] is not None]
        out.append({
            "system": system,
            "runs": len(rs),
            "correct": sum(r["correct"] for r in rs),
            "unsafe": sum(r["unsafe"] for r in rs),
            "none": sum(outcome_of(r) == "none" for r in rs),
            "rescued": sum(r["rescued"] for r in rs),
            "rounds": avg(r["rounds"] for r in rs) if negotiated else None,
            "caught": avg(r["invalid_caught"] for r in rs) if system in ("hybrid", "heuristic") else None,
            "ratio": avg(ratios),
            "extra_travel": avg(extra),
            "calls": avg(r["llm_calls"] for r in rs),
            "tokens": avg(r["input_tokens"] + r["output_tokens"] for r in rs),
            "runtime": avg(r["runtime_s"] for r in rs),
        })
    return out


def write_tables(rows: list[dict], path: Path) -> None:
    lines = ["# Phase 4 results", ""]
    model = next((r["model"] for r in rows if r["model"]), None)
    repeats = max(r["repeat"] for r in rows)
    lines += [f"{len(rows)} runs. LLM model: `{model}`, up to {repeats} repeats per scenario. "
              "Deterministic systems run once.", "",
              "**Correct** means a valid meeting on a solvable scenario, or a proof of impossibility on the "
              "impossible one. Giving up without proof is not correct.", ""]

    lines += ["## By system", "",
              "| System | Runs | Correct | Impossible meeting scheduled | No meeting, nothing proved | "
              "Rescued by fallback | Rounds | Invalid proposals caught | Preference score vs best | "
              "Travel vs best (min) | LLM calls | Tokens | Runtime (s) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in system_summary(rows):
        ratio = None if s["ratio"] is None else s["ratio"] * 100
        lines.append(f"| {SYSTEM_LABELS[s['system']]} | {s['runs']} | {s['correct']} | {s['unsafe']} | {s['none']} | "
                     f"{s['rescued']} | {fmt(s['rounds'])} | {fmt(s['caught'])} | {fmt(ratio, 0, '%')} | "
                     f"{fmt(s['extra_travel'], 0)} | {fmt(s['calls'])} | {fmt(s['tokens'], 0)} | {fmt(s['runtime'], 2)} |")
    lines += ["", "Rounds, calls, tokens and runtime are averages per run. Preference score and extra travel use valid "
                  "meetings only, compared with the solver's best slot. Travel below 0 means less travel than the best slot, "
                  "usually bought by giving up preferences (the solver ranks preferences before travel). LLM runtime "
                  "includes the 7 s spacing between free-tier calls.", ""]

    lines += ["## By tier (correct runs)", "",
              "| Tier | " + " | ".join(SYSTEM_LABELS[s] for s in SYSTEM_ORDER) + " |", "|---|---|---|---|---|"]
    tiers = group(rows, "tier")
    for tier in TIER_ORDER:
        cells = []
        for system in SYSTEM_ORDER:
            rs = [r for r in tiers.get(tier, []) if r["system"] == system]
            cells.append(f"{sum(r['correct'] for r in rs)}/{len(rs)}" if rs else "–")
        lines.append(f"| {tier} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## By scenario", "",
              "| Scenario | Tier | System | Correct | Impossible meeting scheduled | Rounds | LLM calls | "
              "Preference score vs best |", "|---|---|---|---|---|---|---|---|"]
    scenarios = group(rows, "scenario")
    for name in sorted(scenarios):
        by_system = group(scenarios[name], "system")
        for system in SYSTEM_ORDER:
            rs = by_system.get(system)
            if not rs:
                continue
            ratio = avg(score_ratio(r) for r in rs)
            lines.append(f"| {name} | {rs[0]['tier']} | {SYSTEM_LABELS[system]} | {sum(r['correct'] for r in rs)}/{len(rs)} | "
                         f"{sum(r['unsafe'] for r in rs)} | {fmt(avg(r['rounds'] for r in rs))} | "
                         f"{fmt(avg(r['llm_calls'] for r in rs))} | "
                         f"{fmt(None if ratio is None else ratio * 100, 0, '%')} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# --- charts ---------------------------------------------------------------------

def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_axisbelow(True)


def _short(name: str) -> str:
    return name[:2]


def chart_outcomes(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    groups = group(rows, "system")
    systems = [s for s in reversed(SYSTEM_ORDER) if s in groups]
    fig, ax = plt.subplots(figsize=(8.5, 3.4), facecolor=SURFACE)
    for y, system in enumerate(systems):
        rs = groups[system]
        left = 0.0
        for kind in OUTCOME_COLORS:
            n = sum(outcome_of(r) == kind for r in rs)
            share = 100 * n / len(rs)
            if n:
                ax.barh(y, share, left=left, height=0.5, color=OUTCOME_COLORS[kind], edgecolor=SURFACE, linewidth=2,
                        label=OUTCOME_LABELS[kind])
                ax.text(left + share / 2, y, f"{n}/{len(rs)}", ha="center", va="center", color="white",
                        fontsize=10, fontweight="bold")
            left += share
    ax.set_yticks(range(len(systems)), [SYSTEM_LABELS[s] for s in systems], color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of runs (%)", color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    _style(ax)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False,
              labelcolor=MUTED)
    ax.set_title("With the solver as referee, every run ended correctly", loc="left", color=INK, fontsize=13,
                 fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def _grouped(ax, names: list[str], series: dict[str, list[float | None]], fmt_value) -> None:
    width = 0.8 / len(series)
    for i, (system, values) in enumerate(series.items()):
        xs = [x + (i - (len(series) - 1) / 2) * width for x in range(len(names))]
        heights = [v or 0 for v in values]
        bars = ax.bar(xs, heights, width=width, color=SYSTEM_COLORS[system], edgecolor=SURFACE, linewidth=2,
                      label=SYSTEM_LABELS[system])
        for bar, value in zip(bars, values):
            if value is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt_value(value), ha="center",
                        va="bottom", color=MUTED, fontsize=8)
    ax.set_xticks(range(len(names)), [_short(n) for n in names], color=INK)
    ax.set_xlabel("Scenario", color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    _style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(series), frameon=False, labelcolor=MUTED)


def chart_preferences(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    names = sorted({r["scenario"] for r in rows if r["best_score"]})
    series = {}
    for system in ("hybrid", "heuristic"):
        series[system] = []
        for n in names:
            ratio = avg(score_ratio(r) for r in rows if r["scenario"] == n and r["system"] == system)
            series[system].append(None if ratio is None else ratio * 100)
    fig, ax = plt.subplots(figsize=(8.5, 4), facecolor=SURFACE)
    _grouped(ax, names, series, lambda v: f"{v:.0f}")
    ax.set_ylim(0, 118)
    ax.set_ylabel("Preference score, % of solver's best", color=MUTED)
    ax.set_title("Valid is not always best: hybrid can settle below the solver's best slot", loc="left", color=INK,
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def chart_calls(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    names = sorted({r["scenario"] for r in rows})
    series = {s: [avg(r["llm_calls"] for r in rows if r["scenario"] == n and r["system"] == s) for n in names]
              for s in ("hybrid", "llm_only")}
    fig, ax = plt.subplots(figsize=(8.5, 4), facecolor=SURFACE)
    _grouped(ax, names, series, lambda v: f"{v:.0f}")
    ax.set_ylabel("LLM calls per run (average)", color=MUTED)
    ax.margins(y=0.12)
    ax.set_title("LLM calls per negotiation", loc="left", color=INK, fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build tables and charts from the Phase 4 results.")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    rows = load(out_dir / "runs.csv")
    write_tables(rows, out_dir / "summary.md")
    charts = out_dir / "charts"
    charts.mkdir(exist_ok=True)
    chart_outcomes(rows, charts / "outcomes.png")
    chart_preferences(rows, charts / "preferences.png")
    chart_calls(rows, charts / "llm_calls.png")
    print(f"Wrote {out_dir / 'summary.md'} and 3 charts in {charts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
