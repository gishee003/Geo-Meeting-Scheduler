# Phase 4 results

72 runs. LLM model: `gemini-3.5-flash-lite`, up to 3 repeats per scenario. Deterministic systems run once.

**Correct** means a valid meeting on a solvable scenario, or a proof of impossibility on the impossible one. Giving up without proof is not correct.

## By system

| System | Runs | Correct | Impossible meeting scheduled | No meeting, nothing proved | Rescued by fallback | Rounds | Invalid proposals caught | Preference score vs best | Travel vs best (min) | LLM calls | Tokens | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hybrid (ours) | 27 | 27 | 0 | 0 | 0 | 4.0 | 3.0 | 83% | -2 | 6.7 | 9071 | 61.57 |
| LLM only | 27 | 7 | 9 | 11 | 0 | 4.9 | – | 51% | -14 | 12.8 | 12747 | 119.38 |
| Solver only | 9 | 8 | 1 | 0 | 0 | – | – | 100% | 0 | 0.0 | 0 | 0.06 |
| Rule-based agents | 9 | 9 | 0 | 0 | 0 | 3.6 | 2.7 | 95% | 0 | 0.0 | 0 | 0.78 |

Rounds, calls, tokens and runtime are averages per run. Preference score and extra travel use valid meetings only, compared with the solver's best slot. Travel below 0 means less travel than the best slot, usually bought by giving up preferences (the solver ranks preferences before travel). LLM runtime includes the 7 s spacing between free-tier calls.

## By tier (correct runs)

| Tier | Hybrid (ours) | LLM only | Solver only | Rule-based agents |
|---|---|---|---|---|
| easy | 6/6 | 2/6 | 2/2 | 2/2 |
| medium | 9/9 | 2/9 | 3/3 | 3/3 |
| hard | 6/6 | 0/6 | 2/2 | 2/2 |
| impossible | 3/3 | 0/3 | 1/1 | 1/1 |
| plain-language | 3/3 | 3/3 | 0/1 | 1/1 |

## By scenario

| Scenario | Tier | System | Correct | Impossible meeting scheduled | Rounds | LLM calls | Preference score vs best |
|---|---|---|---|---|---|---|---|
| 01_simple_team_sync | easy | Hybrid (ours) | 3/3 | 0 | 4.7 | 6.7 | 100% |
| 01_simple_team_sync | easy | LLM only | 0/3 | 3 | 1.0 | 3.0 | – |
| 01_simple_team_sync | easy | Solver only | 1/1 | 0 | 0.0 | 0.0 | 100% |
| 01_simple_team_sync | easy | Rule-based agents | 1/1 | 0 | 5.0 | 0.0 | 100% |
| 02_busy_week | medium | Hybrid (ours) | 3/3 | 0 | 4.3 | 7.3 | 88% |
| 02_busy_week | medium | LLM only | 2/3 | 0 | 4.0 | 13.3 | 100% |
| 02_busy_week | medium | Solver only | 1/1 | 0 | 0.0 | 0.0 | 100% |
| 02_busy_week | medium | Rule-based agents | 1/1 | 0 | 5.0 | 0.0 | 100% |
| 03_impossible | impossible | Hybrid (ours) | 3/3 | 0 | 0.0 | 0.0 | – |
| 03_impossible | impossible | LLM only | 0/3 | 0 | 8.0 | 17.0 | – |
| 03_impossible | impossible | Solver only | 1/1 | 0 | 0.0 | 0.0 | – |
| 03_impossible | impossible | Rule-based agents | 1/1 | 0 | 0.0 | 0.0 | – |
| 04_same_city_travel | easy | Hybrid (ours) | 3/3 | 0 | 5.0 | 7.0 | 0% |
| 04_same_city_travel | easy | LLM only | 2/3 | 0 | 5.7 | 12.3 | 0% |
| 04_same_city_travel | easy | Solver only | 1/1 | 0 | 0.0 | 0.0 | 100% |
| 04_same_city_travel | easy | Rule-based agents | 1/1 | 0 | 1.0 | 0.0 | 100% |
| 05_multi_city_flights | medium | Hybrid (ours) | 3/3 | 0 | 6.3 | 9.7 | 100% |
| 05_multi_city_flights | medium | LLM only | 0/3 | 0 | 8.0 | 17.7 | – |
| 05_multi_city_flights | medium | Solver only | 1/1 | 0 | 0.0 | 0.0 | 100% |
| 05_multi_city_flights | medium | Rule-based agents | 1/1 | 0 | 5.0 | 0.0 | 100% |
| 06_london_time_zones | medium | Hybrid (ours) | 3/3 | 0 | 5.0 | 7.0 | 100% |
| 06_london_time_zones | medium | LLM only | 0/3 | 3 | 2.0 | 5.0 | – |
| 06_london_time_zones | medium | Solver only | 1/1 | 0 | 0.0 | 0.0 | 100% |
| 06_london_time_zones | medium | Rule-based agents | 1/1 | 0 | 5.0 | 0.0 | 100% |
| 07_four_cities_dense | hard | Hybrid (ours) | 3/3 | 0 | 5.0 | 8.0 | 100% |
| 07_four_cities_dense | hard | LLM only | 0/3 | 0 | 8.0 | 25.0 | – |
| 07_four_cities_dense | hard | Solver only | 1/1 | 0 | 0.0 | 0.0 | 100% |
| 07_four_cities_dense | hard | Rule-based agents | 1/1 | 0 | 5.0 | 0.0 | 100% |
| 08_global_five_people | hard | Hybrid (ours) | 3/3 | 0 | 5.0 | 9.0 | 100% |
| 08_global_five_people | hard | LLM only | 0/3 | 3 | 2.7 | 11.7 | – |
| 08_global_five_people | hard | Solver only | 1/1 | 0 | 0.0 | 0.0 | 100% |
| 08_global_five_people | hard | Rule-based agents | 1/1 | 0 | 5.0 | 0.0 | 100% |
| 09_plain_language_notes | plain-language | Hybrid (ours) | 3/3 | 0 | 1.0 | 6.0 | 80% |
| 09_plain_language_notes | plain-language | LLM only | 3/3 | 0 | 4.7 | 10.3 | 53% |
| 09_plain_language_notes | plain-language | Solver only | 0/1 | 1 | 0.0 | 0.0 | – |
| 09_plain_language_notes | plain-language | Rule-based agents | 1/1 | 0 | 1.0 | 0.0 | 60% |
