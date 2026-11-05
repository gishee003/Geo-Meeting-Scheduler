"""Demo app for the viva.

    .venv\\Scripts\\streamlit run app.py

Pick a scenario and a system, then replay a saved run (works offline) or run a
negotiation live. The Showcase tab puts "AI alone" next to "AI + solver".
"""

import matplotlib.pyplot as plt
import streamlit as st

from scheduler import ui
from scheduler.env import load_dotenv
from scheduler.llm import LLMError

st.set_page_config(page_title="Geo-Aware Meeting Scheduler", page_icon="📅", layout="wide")
load_dotenv(ui.ROOT / ".env")

AVATARS = {"Mediator": "🧭", "Referee": "⚖️"}
SHOWCASE_DEFAULT = "06_london_time_zones"


@st.cache_resource
def all_scenarios():
    return ui.list_scenarios()


@st.cache_resource
def scenario(stem):
    return ui.load(stem)


def show_entry(e):
    with st.chat_message(e.speaker, avatar=AVATARS.get(e.speaker, "🧑‍💼")):
        if e.kind == "referee":
            if e.text.startswith(("INVALID", "NO VALID")):
                st.error(f"**Referee:** {e.text}")
            elif e.text.startswith("VALID"):
                st.success(f"**Referee:** {e.text}")
            else:
                st.info(f"**Referee:** {e.text}")
        else:
            st.markdown(f"**{e.speaker}:** {e.text}")


def show_figure(fig):
    if fig is not None:
        st.pyplot(fig, width="stretch")
        plt.close(fig)


def show_verdict(sc, status, start, venue, compact=False):
    """The solver's final word on a result, checked against what people really need."""
    check = ui.final_check(sc, start, venue)
    if check.verdict is not None:
        slot = ui.slot_text(sc, start, venue)
        if check.verdict.valid:
            st.success(f"✅ **Valid meeting:** {slot}")
        else:
            st.error(f"❌ **Invalid meeting:** {slot}. It cannot actually happen.")
        if not compact:
            st.code(check.explanation, language=None)
    elif status == "impossible":
        st.warning("⚖️ **Proved impossible:** no slot satisfies everyone's must-follow rules.")
    else:
        msg = f"⏹️ **No meeting** (status: {status})."
        if check.best:
            msg += f" Yet a valid slot exists: {ui.slot_text(sc, check.best.start, check.best.venue)}."
        st.warning(msg)
    if check.best and not compact:
        st.info(f"Solver's best slot for comparison: {ui.slot_text(sc, check.best.start, check.best.venue)} "
                f"(preference score {check.best.verdict.score}/{check.max_score})")
    return check


# --- sidebar --------------------------------------------------------------------------

scenarios = all_scenarios()
labels = {stem: f"{stem[:2]} · {name} ({tier})" for stem, name, tier in scenarios}

st.sidebar.title("📅 Meeting scheduler")
stem = st.sidebar.selectbox("Scenario", [s for s, _, _ in scenarios], format_func=labels.get)
mode = st.sidebar.radio("System", list(ui.MODES), format_func=ui.MODES.get)
source = st.sidebar.radio("Source", ["Replay a saved run (works offline)", "Run live now"])

run_path, specs, max_rounds = None, [], 8
if source.startswith("Replay"):
    paths = ui.saved_runs(stem, mode)
    if paths:
        run_path = st.sidebar.selectbox("Saved run", paths, format_func=lambda p: p.stem.split("__")[-1].replace("r", "Run "))
    else:
        st.sidebar.info("No saved run for this choice. Switch to 'Run live now'.")
else:
    if mode in ui.LLM_MODES:
        chosen = st.sidebar.multiselect("AI providers (in fallback order)", list(ui.PROVIDERS),
                                        default=list(ui.PROVIDERS))
        specs = [ui.PROVIDERS[c] for c in chosen]
        st.sidebar.caption("Gemini needs GEMINI_API_KEY in .env. Ollama runs on this laptop.")
    max_rounds = st.sidebar.slider("Maximum rounds", 2, 12, 8)
start_clicked = st.sidebar.button("▶ Start", type="primary", width="stretch",
                                  disabled=source.startswith("Replay") and run_path is None)

sc = scenario(stem)
key = (stem, mode, source, str(run_path))
if st.session_state.get("run", {}).get("key") != key:
    st.session_state.run = {"key": key}
run = st.session_state.run

st.title(sc.name)
st.caption(sc.description)

tab_scn, tab_neg, tab_res, tab_show, tab_p4 = st.tabs(
    ["1 · Scenario", "2 · Negotiation", "3 · Result", "4 · Showcase: AI alone vs AI + solver", "5 · Phase 4 results"])

# --- 1. scenario ------------------------------------------------------------------------

with tab_scn:
    m = sc.meeting
    st.markdown(f"**Meeting:** {m.title}, {m.duration} min, over {sc.horizon_days} days from "
                f"{sc.to_datetime(0):%a %d %b %Y}")
    if sc.geo_enabled:
        st.markdown("**Candidate venues:** " + ", ".join(sc.location_name(v) for v in m.venues))
    st.dataframe(ui.people_rows(sc), width="stretch", hide_index=True)
    st.subheader("Calendars")
    show_figure(ui.calendar_figure(sc))
    if sc.geo_enabled:
        st.subheader("Where everyone is based")
        show_figure(ui.map_figure(sc))

# --- 2. negotiation ---------------------------------------------------------------------

with tab_neg:
    if start_clicked:
        run.update(entries=[], done=False)
        if source.startswith("Replay"):
            row, entries = ui.load_saved(run_path)
            for e in entries:
                show_entry(e)
            start = int(row["start"]) if row["start"] not in (None, "") else None
            run.update(entries=entries, status=row["status"], start=start, venue=row["venue"],
                       rounds=row["rounds"], caught=row["invalid_caught"], calls=row["llm_calls"], done=True)
        else:
            def on_post(e):
                run["entries"].append(e)
                show_entry(e)

            try:
                with st.spinner("Negotiating..."):
                    out, llm = ui.run_live(sc, mode, specs, max_rounds, on_post)
                run.update(status=out.status, start=out.start, venue=out.venue, rounds=out.rounds,
                           caught=out.invalid_caught, calls=llm.usage.calls if llm else 0, done=True,
                           error=out.error)
                if out.error:
                    st.error(out.error)
            except (LLMError, ValueError) as e:
                st.error(str(e))
    elif run.get("done"):
        for e in run["entries"]:
            show_entry(e)
    else:
        st.info("Choose a scenario and system in the sidebar, then press ▶ Start.")
    if run.get("done") and mode == "solver_only":
        st.info("Solver only: there is no conversation. The solver computed the answer directly; see the Result tab.")

# --- 3. result ----------------------------------------------------------------------------

with tab_res:
    if not run.get("done"):
        st.info("Run or replay a negotiation first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Outcome", run["status"])
        c2.metric("Rounds", run["rounds"])
        c3.metric("Invalid proposals caught", run["caught"])
        c4.metric("AI calls", run["calls"])
        check = show_verdict(sc, run["status"], run["start"], run["venue"])
        st.subheader("Calendar with the meeting")
        valid = check.verdict.valid if check.verdict else True
        show_figure(ui.calendar_figure(sc, run["start"], run["venue"], valid))
        if sc.geo_enabled and run["venue"]:
            st.subheader("Travel")
            show_figure(ui.map_figure(sc, run["venue"]))

# --- 4. showcase ----------------------------------------------------------------------------

with tab_show:
    st.markdown("Same scenario, same AI model (Gemini Flash-Lite). **Left:** the AI agents negotiate alone. "
                "**Right:** the same agents, but the solver checks every proposal. Both are saved runs from "
                "the Phase 4 experiment.")
    stems = [s for s, _, _ in scenarios]
    show_stem = st.selectbox("Showcase scenario", stems, index=stems.index(SHOWCASE_DEFAULT),
                             format_func=labels.get, key="showcase")
    ssc = scenario(show_stem)
    left, right = st.columns(2)
    for col, system, title in ((left, "llm_only", "🤖 AI alone"), (right, "hybrid", "🤖 + ⚖️ AI + solver (ours)")):
        with col:
            st.subheader(title)
            paths = ui.saved_runs(show_stem, system)
            if not paths:
                st.info("No saved run.")
                continue
            # Prefer a run that shows the typical failure of AI alone.
            chosen = paths[0]
            if system == "llm_only":
                chosen = next((p for p in paths if ui.load_saved(p)[0]["unsafe"]), paths[0])
            row, entries = ui.load_saved(chosen)
            start = int(row["start"]) if row["start"] not in (None, "") else None
            show_verdict(ssc, row["status"], start, row["venue"], compact=True)
            with st.container(height=520):
                for e in entries:
                    show_entry(e)

# --- 5. phase 4 results --------------------------------------------------------------------

with tab_p4:
    summary = ui.phase4_summary(stem)
    if summary:
        st.markdown(f"**This scenario** across all Phase 4 runs:")
        st.dataframe(summary, width="stretch", hide_index=True)
    st.markdown("**All scenarios** (72 runs; details in `results/summary.md`):")
    charts = ui.RESULTS_DIR / "charts"
    for name in ("outcomes.png", "llm_calls.png", "preferences.png"):
        if (charts / name).exists():
            st.image(str(charts / name), width="stretch")
