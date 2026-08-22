# interventionkit

Record, index and analyse **human interventions** in agent rollouts.

When a person supervises a multi-step agent and takes over to correct it, two
different things get recorded — and they are usually confused with each other:

- **where the supervisor intervened** (a timestamp), and
- **where the agent actually went wrong** (a cause).

They are frequently not the same step. A grasp misaligned during *approach* only
becomes visible when the object slips during *lift*. If the correction is
credited to the phase where the symptom appeared, the cause is never corrected
and the agent keeps making it.

`interventionkit` records both, keeps them separate, and measures the gap.

```bash
pip install interventionkit
```

Dependencies: **numpy only**. It installs cleanly beside any RL or robotics
stack and pulls in no plotting or dataframe libraries.

---

## Recording

```python
from interventionkit import InterventionRecorder

rec = InterventionRecorder(
    "runs/session1",
    task="pick_place",
    phase_names=("approach", "grasp", "lift", "place"),
)

with rec.episode(seed=0, instruction="pick up the red block") as ep:
    while not done:
        if human_has_control:
            ep.human_step(action, phase=phase, proprio=obs["proprio"])
        else:
            ep.policy_step(action, phase=phase, proprio=obs["proprio"])

    # Optional, and the interesting part: ask where the error actually was.
    ep.attribute(phase=0, confidence=0.6, notes="approach looked offset")

    ep.finish(success=False, ground_truth={"root_phase": 0, "root_onset_step": 12})
```

Intervention segments are derived from the per-step actor labels, so segment
boundaries can never disagree with the recorded actions. `policy_step` and
`human_step` are separate methods rather than an `actor="..."` argument
because a typo in a string silently mislabels every correction downstream,
while a misspelled method raises immediately.

## Analysing

```python
from interventionkit import RunReader, analyse

reader = RunReader("runs/session1")
summary = analyse(reader.episodes(), n_phases=4)

summary.onset_misattribution_rate   # intervened in the wrong phase
summary.stated_misattribution_rate  # reported the wrong phase
summary.mean_detection_lag          # steps between true onset and takeover
summary.mean_credit_iou             # overlap of corrected window with the truly-wrong window
summary.late_intervention_rate      # took over after the point of no return
```

Or from the terminal:

```bash
ik-inspect runs/session1 --episodes 5
ik-report  runs/session1 -o report.html
```

`ik-report` writes a **self-contained** HTML page — charts are inline SVG, so it
opens anywhere with no CDN, no JavaScript and no network access.

## What the metrics mean

| metric | question it answers |
|---|---|
| `onset_misattribution_rate` | If you credited the correction to the phase the supervisor took over in — as standard HG-DAgger does — how often would you credit the wrong phase? |
| `stated_misattribution_rate` | If you *asked* them instead, how often are they still wrong? Asking costs interaction time; this says whether it buys accuracy. |
| `mean_detection_lag` | How many steps pass between the error occurring and anyone noticing? |
| `mean_credit_iou` | How much of the correction window actually overlaps the steps that were wrong? This is what predicts downstream policy damage. |
| `late_intervention_rate` | How often is the takeover already too late to rescue the episode? |

Ground truth comes from either controlled fault injection (`root_phase`,
`root_onset_step`) or counterfactual rollout (`pnr_step` — the last step at
which taking over still saves the episode).

## Storage layout

```
runs/session1/
├── run.json          # run-level metadata and config
├── index.jsonl       # one small JSON object per episode
└── episodes/
    ├── ep_000000.npz # arrays: action, actor, phase, + whatever you passed
    └── ep_000000.json
```

Built for machines with limited RAM. `index.jsonl` lets you compute statistics
over tens of thousands of episodes while touching only a few megabytes, and
`np.load` decompresses `.npz` members lazily, so reading just the actions never
materialises the images. Writes are atomic (temp file + rename), and an episode
that raises mid-collection is discarded rather than left truncated.

## Schema stability

Every record carries `schema_version`. A reader refuses a version it does not
understand instead of silently misreading fields — a dataset that loads but
means something different is more expensive than one that fails loudly.

## Status

Alpha. The API above is what the [interactive-correction-for-manipulation](https://github.com/abyyworld/Interactive-correction-for-manipulation)
project uses for its own experiments, so it gets exercised on real runs.
Issues and PRs welcome, especially from anyone applying it to non-robotics
agents — the abstraction is deliberately not robot-specific.

MIT licensed.
