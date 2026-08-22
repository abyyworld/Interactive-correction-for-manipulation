# CLAUDE.md

Orientation for an AI assistant (or a returning human) picking this repository up
cold. Read this before changing anything.

## What this project is

A MuJoCo Franka Panda does a four-phase pick-and-place. A synthetic supervisor
watches, detects failures and takes over. The takeover is recorded as a
correction, the episode is **rewound to a chosen point** and re-demonstrated, and
a behaviour-cloning policy is trained on the result.

The whole repository exists to answer one question: **where should you rewind
to?** Four strategies are compared — `onset` (the takeover step), `symptom`
(where the failure first became visible), `stated` (the phase the supervisor
blamed) and `oracle` (the true cause).

## The result — do not "fix" it

With initial-state coverage held fixed, over 250 evaluation episodes each:

| strategy | rewind depth | success |
|---|---|---|
| `symptom` | 6.0 steps | **42.0%** |
| `onset` | 0.0 steps | 34.4% |
| `stated` | 33.1 steps | 10.8% |
| `oracle` | 34.3 steps | **4.4%** |

**`oracle` — perfect knowledge of the true cause — is the worst strategy.** This
is the finding, not a bug. The rewind is an exact state restore, so rewinding
past the failure lets the corrective expert fly a clean trajectory; the episode
never enters the state the failure produced, and the dataset ends up with no
supervision for it. Correcting the cause deletes the evidence.

Two things follow, and both have caught people out already:

- **Do not "improve" the numbers.** They are measurements. If a change moves
  them, the change altered the experiment.
- **Do not trust offline proxies here.** `oracle` has the *highest* correction
  success rate (91.5%) and the *lowest* validation loss (0.0874). Both rank the
  strategies backwards, because the protocol changes which states are in the
  dataset and the held-out set moves with it.

`results/` holds the raw JSON behind every number quoted in `README.md` and
`docs/study-design.md`. **Never quote a number that is not in `results/`.** If
you need a new number, run the experiment and commit the report.

## Layout

```
src/icm/
  envs/          scene generation, the pick-place task, fault injection
  control/       the scripted expert (100/100) and IK
  study/         synthetic supervisor, attribution study, degradation experiment
  training/      rewind-and-redemonstrate DAgger, dataset, BC trainer
  policies/      the BC policy and its rollout wrapper
  eval/          metrics, rollout, figures
  teleop/        VR/keyboard teleoperation over UDP
  cli/           entry points (icm-study, icm-dagger, icm-train, ...)
interventionkit/  the recording + analysis layer, extracted, numpy-only
results/          committed JSON reports; the source of every quoted number
docs/             study-design.md is the full experimental write-up
```

## Commands

`make help` lists everything. The ones that matter:

```bash
make install                 # numpy + mujoco + pytest only
make assets                  # Panda meshes, pinned commit, ~33 MB
make test                    # 86 tests, ~2 min
make lint
make torch-cpu               # PyTorch is an optional extra, not a dependency
make study                   # attribution study
make degradation-controlled  # the experiment the conclusions rest on (~1 h)
```

## Landmines

Each of these cost real time. They are not theoretical.

### MuJoCo

- **`<compiler>` is last-wins.** The mesh-directory override must come *after*
  the Menagerie include or it is silently ignored.
- **Gravity compensation must be set before compile.** `ngravcomp` is counted at
  compile time; writing `body_gravcomp` afterwards does nothing at all. Set it
  through `MjSpec`. Symptom when wrong: several mm of TCP droop.
- **The upstream keyframe has 9-entry qpos.** Adding free-jointed objects raises
  `nq`, so the keyframe is stripped in a cached copy of the model.
- **`MjSpec` body lookup differs by version** — `find_body(name)` in 3.2.x became
  `body(name)` in 3.3.x. Always use `find_spec_body()` in `envs/assets/scene.py`.
  `pyproject` declares `>=3.2,<3.4`, and CI runs both ends.
- **Create renderers once and reuse them.** Multiple live OSMesa renderers return
  black frames.
- **The GL backend is chosen at import time** in `src/icm/__init__.py`, gated on
  whether a real GPU exists — Mesa provides EGL with no GPU behind it, and its
  destructor then raises `EGLError` on every teardown. Do not force a backend.

### Training

- **MuJoCo and PyTorch segfault together on CPU** when an optimiser steps in a
  process where both are loaded. Training therefore runs in a **subprocess that
  never imports MuJoCo** (`study/degradation.py`). Do not "simplify" this away.
- **`--cap-per-root` is positional.** The corrections root must come first and
  the shared demo pool second. Getting this wrong silently discards the control.
- **The cap subsamples frames uniformly**, so every episode still contributes.
  Truncating instead would drop whole episodes and reintroduce a coverage
  difference through the back door.

### Experiment integrity

- **The supervisor must never read the injected fault.** It detects from
  observable state only. A test parses the AST of `SyntheticSupervisor._detect`
  to enforce this. If that test fails, the experiment is invalid — fix the code,
  never the test.
- **Collection is reused when a prior run exists.** It is seed-determined and
  slow (~20 min). Appending into the same directory would silently double every
  dataset.

### Tooling

- **Run tests as `pytest`, not `python -m pytest`.** `pythonpath = ["."]` in
  `pyproject.toml` makes both work; the Makefile and CI deliberately use the same
  form. They diverged once and CI collected zero tests for 14 consecutive runs
  while passing locally.
- **PyTorch is optional.** Tests needing it are marked `requires_torch`. The
  3.3.7 CI leg installs no torch on purpose, to keep that true.

## Open next steps

1. **Replace the scripted expert with a learned policy as the supervised agent.**
   Currently the thing being corrected is a scripted controller with an injected
   fault, so the ground-truth cause is known exactly — which is the point, but
   real policies fail in messier, correlated ways. This is the biggest limitation
   and the most valuable next experiment.
2. **Sweep small rewind depths.** `symptom` beats `onset` by 7.6 points at
   *p* = 0.08 — suggestive, unresolved. A sweep over depths 0–15 would settle
   whether there is a sweet spot just before the takeover.
3. **A protocol that keeps the cause *and* the failure states** — reweighting
   existing frames rather than replacing them, for instance. Every conclusion
   here is about rewind-and-redemonstrate specifically; that no protocol can use
   a correct attribution is not claimed.

## Working agreement

Four separate times in this project something looked correct and was not: an
edit that silently did not apply, a loss reported at 7× its true value, a
control that was collected and then discarded, and a CI suite that never ran a
single assertion. All four passed casual inspection.

So: **run it before claiming it works.** Check the artefact, not the intention —
the recorded config, the actual command, the CI conclusion. Where a fix is
subtle, add the test that would have caught it. Several of the tests in this repo
exist for exactly that reason.
