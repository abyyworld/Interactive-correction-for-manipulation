# Interactive correction for manipulation

<p align="center">
  <img src="docs/media/pick_place.gif" alt="Franka Panda picking a red block and placing it on the goal pad" width="520">
</p>

A Franka Panda in MuJoCo, a supervisor who intervenes when it goes wrong, and a
measurement of something that interactive imitation learning normally assumes
away:

> **When a human intervenes at step 3 of a multi-step task, was the error
> actually at step 1 — and what does getting that wrong cost the trained policy?**

Standard human-gated DAgger relabels from the moment the human takes the
controls. That is correct only when the mistake *is* where they took over. A
grasp misaligned during **approach** looks fine until the object slips during
**lift**; credit the correction to the lift and the cause is never fixed.

This repository measures how often that happens, and what it costs.

---

## Results

Measured in this repository, reproducible with `make study`. **The supervisor is
simulated** — see [Honest limitations](#honest-limitations).

**Misattribution depends on when the failure becomes *visible*, not on when it was
caused.** Rate at which the phase a failure appears in differs from the phase
that caused it:

| fault | cause → symptom | misattributed | detection lag |
|---|---|---|---|
| `lift_slip` | lift → lift | **0.0%** | 9.9 steps |
| `weak_grip` | grasp → lift | **100.0%** | 23.7 steps |

These two produce *visually identical* failures — the object falls out during
the lift. Only the true cause differs, so the gap cannot be explained by the
failures looking different. n=36 each.

Across all six fault types, grouped by how far the symptom is from the cause
(n=280 episodes, supervisor tracing accuracy 0.35):

| lag class | n | onset | **symptom** | stated | detection lag |
|---|---|---|---|---|---|
| immediate | 74 | 1.000 | **0.000** | 0.000 | 9.8 |
| short | 39 | 0.872 | **0.667** | 0.359 | 33.7 |
| delayed | 75 | 0.813 | **0.893** | 0.507 | 48.2 |
| very delayed | 37 | 0.000 | **0.000** | 0.000 | 48.4 |

Asking the supervisor to name a cause roughly halves the error on delayed faults
(0.893 → 0.507) but does not remove it.

**An unplanned finding.** `wrong_object` (the robot picks the blue block instead
of the red one) has the largest gap between cause and consequence and was
predicted to be the worst case. It is attributed **perfectly — 0.000 across all
three measures** — because picking the wrong block is visible immediately, even
though its consequence only materialises three phases later. *The delay that
drives misattribution is observational, not causal.* The "very delayed" row
above is that finding: a category defined by causal distance that behaves
exactly like the immediate one.

**Supporting numbers**

| quantity | value |
|---|---|
| scripted expert baseline | 100/100 episodes |
| IK accuracy over 200 workspace poses | 200/200 converged, max error 0.2 mm |
| fault detection by the supervisor | 90–97% per fault type |
| episodes rescued by correction | 0% → 53–95% depending on fault |
| unnecessary interventions on 40 healthy episodes | **0** |
| study size | 280 episodes, 37,611 steps |

---

## What is actually built

```
scene + IK ──► scripted expert ──► fault injection ──► supervisor ──► recorder
   100%           100/100           6 fault types      detects 20/20    ↓
                                                                   corrections
                                                                        ↓
                        evaluation ◄── BC policy ◄── rewind + credit assignment
```

* **Environment** — Franka Panda, RGB-D cameras (wrist + scene), four labelled
  phases, and *bit-exact* state snapshot/restore.
* **Scripted expert** — 100% success. Not the deliverable; the instrument.
* **Fault injection** — six faults with known root-cause phase, spanning
  immediate to very delayed consequences.
* **Synthetic supervisor** — detects failures from observable state only, with a
  reaction delay and a tunable ability to trace causes.
* **interventionkit** — the recording and analysis layer, extracted as a
  [standalone package](interventionkit/) with numpy as its only dependency.
* **Rewind-and-redemonstrate DAgger** — the mechanism that turns a stated
  attribution into training data.
* **Behaviour cloning** — action chunking, spatial-softmax visual encoders,
  optional language conditioning.
* **VR teleoperation** — clutched relative control over a UDP protocol, so the
  headset runs in its own process on its own machine.

---

## Quick start

**Requires Python 3.10–3.13.** MuJoCo publishes no wheel for 3.14 yet, so pip
would try to build it from source and fail. `make` picks a suitable interpreter
automatically if one is installed.

```bash
git clone https://github.com/abyyworld/Interactive-correction-for-manipulation
cd Interactive-correction-for-manipulation
make install      # venv + project. numpy, mujoco, pytest. That is all.
make assets       # Panda meshes, pinned commit, ~33 MB
make test         # 71 tests
make study        # the attribution study + an HTML report
```

That is the whole minimum setup: **no GPU, no PyTorch, no plotting libraries**.
The attribution study renders no pixels, so it runs on a laptop in a few
minutes.

Add the optional pieces only when you need them:

```bash
make viz          # matplotlib + imageio, for figures and GIFs
make torch-cpu    # policy training on CPU (small download)
make torch-cuda   # policy training on NVIDIA (~2.5 GB, do this on wifi)
```

Then the learning half:

```bash
make demos EPISODES=800
make train
make eval
```

Run `make help` for everything else.

---

<p align="center">
  <img src="docs/media/misattribution.png" alt="Misattribution by lag class" width="640"><br>
  <img src="docs/media/trace_sweep.png" alt="Misattribution against supervisor tracing accuracy" width="470">
</p>

Full experimental design, including the controls and what it cannot tell you:
**[docs/study-design.md](docs/study-design.md)**.

## The experiment

### Measuring attribution

Errors are **injected deliberately** at a known phase, because on organically
failing rollouts nobody knows the true cause — that is the thing under study.

| fault | cause | symptom | lag |
|---|---|---|---|
| `grasp_offset` | approach | lift | delayed |
| `wrong_object` | approach | place | very delayed |
| `weak_grip` | grasp | lift | delayed |
| `premature_close` | approach | grasp | short |
| `lift_slip` | lift | lift | immediate |
| `early_release` | place | place | immediate |

The last two are controls. If misattribution were an artefact of the
measurement, they would be misattributed as often as the delayed ones.

Three attributions are recorded separately, and keeping them apart matters:

- **onset** — the phase at takeover. What HG-DAgger credits today.
- **symptom** — the phase where the failure became visible. The best a
  timestamp-based scheme can do.
- **stated** — what the supervisor says caused it. The only one that can point
  backwards past the symptom.

Measured by onset alone the result is *backwards* — immediate faults appear
100% misattributed and delayed ones 77%. The reason is that after a dropped
object the phase tracker correctly reverts to `approach`, so for approach-caused
faults the takeover phase accidentally matches. By symptom phase the picture
inverts and makes sense.

### Measuring the cost

When the supervisor names a phase, the episode is **rewound** to the start of it
using an exact state snapshot, and the correction is demonstrated from there.
Only the rewind target differs between conditions:

| strategy | rewinds to | |
|---|---|---|
| `onset` | the takeover step | today's HG-DAgger |
| `symptom` | where the failure became visible | timestamp ceiling |
| `stated` | the phase the supervisor blamed | what asking buys you |
| `oracle` | the true cause | not implementable; the ceiling |

A wrong attribution rewinds to states that were already fine, so the causal
states never receive a corrective action. The gap between `stated` and `oracle`
is the price of misattribution, in units of task success.

Rewinding further necessarily yields *more* corrective frames, so every dataset
is subsampled to the size of the smallest. Otherwise "corrected the right
states" cannot be separated from "had more data".

```bash
make degradation
```

---

## Honest limitations

Stated plainly, because the difference matters:

1. **The supervisor is simulated, not human.** `trace_accuracy` is a free
   parameter, not a measurement. The deliverable is the *curve* across it, so
   that when a VR study with real participants provides a number, the
   corresponding cost can be read off. No result here is a human result.
2. **The agent under supervision is a scripted expert with an injected fault**,
   standing in for a policy with a systematic error, because its failure mode is
   known exactly. A learned policy fails in messier ways.
3. **Behaviour cloning from ~200 demonstrations does not solve this task.**
   Prediction error is good (0.078 mean L1, better than a 1-NN bound of 0.116)
   but execution compounds error and success stays low. This is the well-known
   BC distribution-shift problem, and it is the reason interactive correction
   exists — but it does mean absolute policy numbers here are weak.
4. **Simulation only.** No real robot, no sim-to-real claim.
5. **Language conditioning is templated**, not free-form.

---

## Repository layout

```
src/icm/
  envs/        scene generation, Panda wrapper, phases, fault injection
  control/     damped least-squares IK, scripted expert
  teleop/      protocol, VR receiver, keyboard, base interface
  policies/    behaviour cloning, action chunking, policy runner
  training/    streaming dataset, trainer, DAgger rewind, credit assignment
  study/       synthetic supervisor, attribution study, degradation experiment
  eval/        rollout loop, Wilson-interval metrics
  cli/         icm-collect, icm-train, icm-eval, icm-study, icm-dagger, icm-teleop
interventionkit/   standalone package: recording + attribution analysis
docs/
  study-design.md      the experiment, its controls, and its limits
  remote-training-box.md  running this on a GPU box over SSH
```

## Things that were not obvious

A few findings that cost real debugging time and are documented in the commits:

- **Gravity compensation must be set before compile.** MuJoCo counts
  compensated bodies at compile time, so writing `model.body_gravcomp` on a
  compiled model changes the field and produces no force. It was worth 5.7 mm of
  TCP droop — a quarter of the grasp margin on a 4.2 cm cube.
- **Contact-based grasp detection flickers.** A finger separates for single
  steps under lift acceleration while the cube is plainly held. Unfiltered it
  corrupts phase labels and fires "object dropped" on healthy episodes.
- **A "weak grip" is not a wider gripper opening.** Commanding a narrower
  opening grips a cube *harder*. It has to be modelled as reduced grip force.
- **Adding a lateral bias to a closed-loop controller does nothing.** An earlier
  drift fault was corrected away every step; 15/15 episodes still succeeded. A
  fault must corrupt the controller's *intent*.
- **MuJoCo and PyTorch's MKL kernels segfault together on CPU** when an
  optimiser steps in a process where both are loaded. Training runs in its own
  process, which is a cleaner stage boundary anyway.

## License

MIT.

## Platform support

Tested on Linux (including WSL2), macOS and Windows. The rendering backend is
selected at import time — EGL when a GPU is present, OSMesa on a headless Linux
box, and the platform default on macOS and Windows. Set `MUJOCO_GL` to override.

| platform | notes |
|---|---|
| **Linux / WSL2** | `sudo apt install libosmesa6 libgl1 libegl1` first. With a GPU, EGL is selected automatically and renders ~50× faster. |
| **macOS** (Intel or Apple silicon) | Works out of the box; no GL packages needed. PyTorch uses MPS. |
| **Windows** | Works natively. If you do not have `make`, call the CLIs directly (below). |

Every `make` target is a thin wrapper, so nothing needs `make`:

```bash
python -m icm.envs.assets.fetch                      # make assets
python -m pytest -q                                  # make test
python -m icm.cli.study -o runs/study -n 40 --report # make study
python -m icm.cli.collect -o runs/demos -n 800       # make demos
python -m icm.cli.train runs/demos -o runs/bc        # make train
python -m icm.cli.evaluate --checkpoint runs/bc/checkpoint.pt -n 100
```
