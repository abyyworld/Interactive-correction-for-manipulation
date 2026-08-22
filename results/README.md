# Recorded results

The raw JSON behind every number quoted in the top-level README and in
`docs/study-design.md`. Committed so the claims are auditable without a rerun —
each file is the verbatim output of the command named below.

| file | produced by | what it holds |
|---|---|---|
| `summary.json` | `make study` | attribution study, n=280 episodes |
| `sweep.json` | `make sweep` | misattribution across supervisor tracing accuracy |
| `degradation.json` | `make degradation` | credit assignment, uncontrolled |
| `degradation_controlled.json` | `make degradation-controlled` | credit assignment, coverage held fixed |

`degradation_controlled.json` is the one the conclusions rest on: it adds a
shared demonstration pool so that initial-state coverage is identical across
conditions. Its uncontrolled counterpart is kept because the difference between
the two *is* the finding — the ordering inverts once the confound is removed.

Episode-level recordings are not committed; they run to gigabytes. Regenerate
them with the commands above.
