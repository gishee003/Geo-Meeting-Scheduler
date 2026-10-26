# Project Plan: Geo-Aware Multi-Agent Meeting Scheduler

**Course:** AID 851 – Machine Reasoning
**Topic:** Multi-agent negotiation and planning system
**Deadline:** 27 Nov 2026 · **Viva:** 30 Nov – 4 Dec 2026
**Deliverables:** Working solution + report (goals, methodology, data, facts & constraints, outcome, conclusions)

---

## 1. What we are building (in simple terms)

Every person in a meeting gets an **AI agent** that knows their calendar, their location and their preferences. The agents **negotiate in plain English** to find a time and place that suits everyone.

AI agents can be confidently wrong, so each proposal is checked by a **logic solver (Z3)**. The solver only accepts a proposal that satisfies every hard rule, and when it rejects one it explains the reason.

> **The AI negotiates, and the solver proves the answer is correct.** (Neuro-symbolic reasoning.)

### Geographical extension (from the professor's feedback)
Scheduling also has to consider **where** people are:
- **Travel time:** there must be enough time to travel from the previous event to the meeting venue.
- **Flights:** when someone has to fly, a real flight must arrive early enough before the meeting.
- **Time zones:** preferences such as "mornings only" apply in each person's local time.
- **Venue choice:** the system chooses where to meet (an office in some city, or online).

### Two kinds of rules
| Kind | Meaning | Examples | Enforced by |
|---|---|---|---|
| **Hard constraint** | Must never be broken | No double-booking, enough travel time, flight lands before the meeting, within working hours | Solver (valid / invalid) |
| **Soft preference** | Nice to have | Prefer mornings, avoid Fridays, prefer not to fly | Agents negotiate; the solver scores options |

---

## 2. Architecture

```
  Agent (Person A) ─┐
  Agent (Person B) ─┼──► Negotiation (plain English) ──► Proposal {time, venue}
  Agent (Person C) ─┘                                        │
        ▲                                                    ▼
        │                                         ┌──────────────────────┐
        │  "Invalid: B can't reach Mumbai before  │   Z3 SOLVER          │
        └── 14:00 — earliest flight lands 13:10"  │ calendars, travel,   │
              or "Valid ✓ — meeting confirmed"   │ flights, time zones  │
                                                  └──────────────────────┘
```

**Components**
1. **Data layer:** people, calendars, preferences, locations, travel-time matrix, flight schedule, time zones.
2. **Solver (Z3):** does three jobs.
   - *Verify* a proposed `{time, venue}`.
   - *Explain* why a proposal fails (unsat core translated into plain language).
   - *Suggest* the best valid options (Z3 `Optimize` with soft preferences as weights).
3. **Agents (LLM):** one agent per participant, plus an optional mediator. Each agent sees only its own person's data.
4. **Negotiation loop:** propose → verify → explain → revise, until the solver confirms a valid slot or proves that none exists.
5. **UI (Streamlit):** negotiation transcript, calendar view, solver verdicts, optional map.

**Tech stack:** Python 3, `z3-solver`, an LLM API, `zoneinfo`, Streamlit, pytest.

---

## 3. Phases

### Phase 1: Foundations and core solver (no geography)
**Goal:** given calendars and preferences, the solver alone finds the best valid meeting slot.
- Set up the project structure, virtual environment and dependencies.
- Define the data model in JSON: people, calendar events, working hours, hard constraints and soft preferences.
- Create a few hand-written test scenarios, including one that is impossible on purpose.
- Encode the hard constraints in Z3: no overlap with existing events, within working hours, meeting duration.
- Encode the soft preferences with Z3 `Optimize` and weights.
- Add three solver functions: `verify(proposal)`, `explain(proposal)` and `suggest(k)`.
- Write unit tests.

**Done when:** the solver returns the best slot for a simple scenario, rejects an invalid proposal with a clear reason, and reports "no solution" for the impossible scenario.

### Phase 2: Geographical constraints
**Goal:** the solver understands locations, travel, flights and time zones.
- Add locations: cities and offices with coordinates, plus an "online" option.
- Build a travel-time model: start with an estimate (distance ÷ speed, plus a buffer). Real routing data can be added later, cached.
- Build a flight timetable as a static dataset (based on real routes) so that demos never depend on the internet.
- Handle time zones: store everything in UTC and evaluate preferences in local time.
- Add solver constraints:
  - Travel buffer between back-to-back events at different locations.
  - Flight arrival plus airport-to-venue time comes before the meeting starts.
  - The venue is a decision variable chosen by the solver.
  - Optional objective: minimise total or maximum travel (fairness).
- Make explanations geo-aware, e.g. "Asha needs 45 min to reach the venue; her previous meeting ends at 11:00."

**Done when:** the solver rejects a slot that is free on everyone's calendar but physically unreachable, and picks a sensible venue.

### Phase 3: LLM agents and negotiation
**Goal:** agents negotiate in natural language and every proposal is verified.
- Write an agent prompt per participant that knows only that person's data (privacy).
- Define a structured proposal format (JSON `{start, venue, reason}`) alongside the chat message.
- Run the negotiation loop with a mediator or round-robin turns.
- Send solver feedback to agents as plain-language explanations.
- Add safeguards: a maximum number of rounds, a solver suggestion when agents are stuck, and the ability to declare "no valid slot exists".
- Log full transcripts for the report.

**Done when:** a multi-city scenario with three or more people runs end to end and ends in a solver-verified meeting.

### Phase 4: Evaluation
**Goal:** show with data why the hybrid approach is better.
- Compare three systems on the same scenarios:
  1. **LLM only** (no solver)
  2. **Solver only** (no negotiation)
  3. **Hybrid** (our system)
- Measure:
  - Validity rate of final answers
  - Negotiation rounds
  - Preference satisfaction score
  - Total travel time
  - LLM calls and cost
  - Runtime
- Use scenario tiers (easy, medium, hard, impossible) that vary the number of people, cities, calendar density and flights.
- Produce result tables and charts.

**Done when:** results are reproducible with one script and charts are generated.

### Phase 5: Demo UI
**Goal:** a clear demo for the viva.
- Streamlit app with:
  - Scenario picker
  - Live negotiation transcript
  - Solver verdicts
  - Final calendar view
  - Optional map of participants and venue
- One "showcase" scenario where the LLM alone fails and the hybrid succeeds.

### Phase 6: Report and viva preparation
- Report sections: Goals · Methodology · Data · Facts & Constraints · Outcome · Conclusions, plus Limitations and Future Work.
- Include an architecture diagram, constraint formulations, example transcripts and evaluation results.
- Prepare viva slides and rehearse the demo. Expect questions like "why a solver?" and "how is travel time encoded?".

---

## 4. Risks and mitigations
| Risk | Mitigation |
|---|---|
| LLM API cost or instability | Use a small model while developing, cache responses, limit rounds |
| Flight or map APIs unavailable | Use static, cached datasets; the demo works offline |
| LLM output not parseable | Structured JSON output, validate it, retry |
| Scope creep | Treat extras (live maps, rescheduling, recurring meetings) as future work |

---

## 5. Progress tracker
- [x] Phase 1: Foundations and core solver *(done 26 Oct 2026: Z3 verify/suggest/diagnose/explain, 3 scenarios, 14 tests incl. brute-force cross-check)*
- [ ] Phase 2: Geographical constraints
- [ ] Phase 3: LLM agents and negotiation
- [ ] Phase 4: Evaluation
- [ ] Phase 5: Demo UI
- [ ] Phase 6: Report and viva preparation
