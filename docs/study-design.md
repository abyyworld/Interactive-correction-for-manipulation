# Study design

How the two experiments in this repository are constructed, and why each choice
was made. The intended reader is someone deciding whether to believe the result.

---

## The question

In a multi-step task, an agent's mistake and its visible consequence can occur in
different phases. A grasp misaligned during **approach** produces no visible
problem until the object slips during **lift**.

Human-gated DAgger treats the moment of takeover as the moment of error: it
relabels from that timestamp onward and leaves everything before it alone. Two
things follow, and this repository measures both.

1. **How often does a supervisor attribute an error to the wrong phase?**
2. **What does that cost the policy trained on their corrections?**

---

## Experiment 1: attribution

### Why errors are injected rather than observed

On an organically failing rollout, nobody knows the true cause. "The policy went
wrong here" is an interpretation, and interpretation is exactly what is under
study — using it as ground truth would be circular.

So a fault is introduced **deliberately**, at a chosen phase, with a recorded
onset step. The supervisor is never told. Their attribution is then scored
against a fact rather than against another opinion.

### The fault set

| fault | cause | symptom | causal lag |
|---|---|---|---|
| `grasp_offset` | approach | lift | delayed |
| `wrong_object` | approach | place | very delayed |
| `weak_grip` | grasp | lift | delayed |
| `premature_close` | approach | grasp | short |
| `lift_slip` | lift | lift | immediate |
| `early_release` | place | place | immediate |

`lift_slip` and `early_release` are **controls**. If misattribution were an
artefact of the measurement rather than a real effect, they would be
misattributed as often as the delayed faults.

`weak_grip` and `lift_slip` are a **matched pair** and carry most of the
argument. Both end with the object falling out during the lift, so the symptom
is visually identical; only the true cause differs. A difference in attribution
between them cannot be explained by the failures looking different — the obvious
confound for this kind of study.

### Faults corrupt intent, not output

An early version added a lateral velocity bias during the lift. It had no effect
at all: the expert closes the loop on measured TCP position every step and
corrected the bias away, with 15/15 episodes still succeeding.

A fault has to corrupt what the controller is *trying to do* — its grasp target,
its chosen object, its grip force — because that is what a mis-generalising
policy actually looks like. Additive noise is recovered from by the next
timestep and produces no delayed consequence to misattribute.

### Faults are suspended on takeover

A fault models the *agent's* error, not a broken robot. If it persisted through
the handover, the human's correction would fail too, and the correction data
would describe an unfixable robot rather than a fixable policy.

### The supervisor

Detection uses only what a person watching the screen could see — an empty
gripper rising, an object dropping, the wrong block in the hand, an object
knocked away, or nothing happening at all. It never reads the injected fault;
this is asserted structurally by a test that parses the detector's AST and fails
if it references any privileged name.

A fault whose symptom is subtle is therefore detected late *by construction*,
which is the effect under study rather than something hard-coded.

Every predicate had to survive healthy episodes without firing. Early versions
fired on 65% of them — a normal release at the goal looks exactly like a dropped
object on speed alone, and a healthy approach legitimately runs 76 steps, so
thresholding stall on phase duration manufactured interventions where nothing
was wrong. False positives are not harmless extra data: they record an
intervention with no root cause. The rate is now **0/40**.

### Three attributions, kept separate

| name | what it is | who would use it |
|---|---|---|
| **onset** | phase at takeover | HG-DAgger today |
| **symptom** | phase when the failure became visible | best possible from a timestamp |
| **stated** | the phase the supervisor names | a system that asks |

Separating them was necessary, not tidy. Measured by onset alone, immediate
faults appear **100%** misattributed and delayed ones **77%** — backwards. The
cause: after a dropped object the phase tracker correctly reverts to `approach`,
so for approach-caused faults the takeover phase *accidentally* matches the root
cause. Measured by symptom phase the picture inverts and makes sense.

This is worth stating because the naive measurement does not merely add noise —
it reverses the sign of the effect.

---

## Experiment 2: what misattribution costs

### Rewind and re-demonstrate

When the supervisor names a phase, the episode is **rewound** to the start of
that phase using an exact state snapshot, and the correction is demonstrated
from there. The recorded episode is policy actions up to the rewind point and
corrective actions after it.

| strategy | rewinds to |
|---|---|
| `onset` | the takeover step (no rewind) |
| `symptom` | where the failure became visible |
| `stated` | the start of the phase the supervisor blamed |
| `oracle` | the true root-cause onset |

`oracle` is not implementable in practice. It is the ceiling the others are
measured against, and the gap between `stated` and `oracle` at a given tracing
accuracy is the price of misattribution in units of task success.

This is the only reason the environment guarantees **bit-exact** state restore.
It is not a convenience feature.

### What the experiment actually found

The prediction failed twice, in two different directions, and the second failure
is the result worth keeping.

**First pass.** `stated` (rewind to the phase the supervisor blamed) and `oracle`
(rewind to the true cause) came out **statistically indistinguishable**: 22.0%
versus 20.8% over 250 evaluation episodes, intervals overlapping. Meanwhile
`onset` — the naive strategy with no rewind at all — reached 69.6%, far ahead of
every rewinding condition.

The covariate that explained that ranking was not attribution accuracy but where
the corrective demonstration *began*:

| strategy | corrections starting in APPROACH | success |
|---|---|---|
| `onset` | 87.8% | 69.6% |
| `oracle` | 41.0% | 20.8% |
| `stated` | 34.6% | 22.0% |
| `symptom` | 27.1% | 11.2% |

Every evaluation episode starts in APPROACH. By the time the supervisor reacts,
the dropped object has settled and the phase tracker has correctly reverted to
APPROACH, so the naive rewind happens to produce corrections that begin at the
start of the task. That is a confound: the conditions differed in how much of
the task their data covered, not only in where they assigned credit.

**Second pass, with the confound removed.** Every condition now trains on one
shared pool of 150 clean demonstrations (11,132 frames), collected once and
reused byte-for-byte, plus its own corrections subsampled to a common budget of
10,558 frames. Initial-state coverage is identical by construction.

| strategy | rewinds to | mean rewind depth | success (n=250) |
|---|---|---|---|
| `symptom` | first visible | 6.0 steps | **42.0%** [36.0, 48.2] |
| `onset` | takeover step | 0.0 steps | 34.4% [28.8, 40.5] |
| `stated` | blamed phase | 33.1 steps | 10.8% [7.5, 15.3] |
| `oracle` | true cause | 34.3 steps | 4.4% [2.5, 7.7] |

The ordering inverts. `symptom` moves from worst to best. `oracle` becomes the
worst of the four, losing to `stated` — its own noisy approximation — by 6.4
points (*p* = 0.007), and to `onset` by 30.0 points (*p* = 2 × 10⁻¹⁷).

The surviving covariate is rewind depth, and the mechanism is mechanical. The
rewind is an exact state restore, so the corrective expert takes control at the
rewind point and flies from there. Rewind 34 steps and the arm is still in a
state that looks healthy; the expert drives a clean trajectory out of it and the
episode never enters the state the failure would have produced. The dataset
therefore contains **no supervision for the failure at all** — it contains
another demonstration. Rewind to the takeover instead, and the first frame of
the correction *is* the failure state.

Correcting the cause deletes the evidence.

### Two proxies for correction quality both invert the answer

| strategy | corrections that succeed | best val loss | policy success |
|---|---|---|---|
| `symptom` | 84.5% | 0.0878 | **42.0%** |
| `onset` | 86.0% | 0.1013 | 34.4% |
| `stated` | 88.0% | 0.0895 | 10.8% |
| `oracle` | **91.5%** | **0.0874** | **4.4%** |

`oracle` produces the highest-quality corrections by both available offline
measures and the worst policy by the only measure that matters. Validation loss
is computed on the corrective data's own distribution; the policy is evaluated on
the distribution it induces itself. When a protocol changes *which states* end up
in the dataset, held-out loss stops being a proxy for anything, because the
held-out set moved too.

This is worth stating plainly because it is the trap the experiment nearly fell
into: had the strategies been ranked by validation loss, or by how often the
corrections themselves succeeded, the conclusion would have been the exact
opposite of the truth.

### What is not resolved

`symptom` versus `onset` is +7.6 points at *p* = 0.08 — suggestive, not resolved.
Rewinding a few steps before the takeover is plausibly the sweet spot: far enough
back to catch the failure as it forms, not so far as to step over it. Separating
those two would need either more evaluation episodes or a deliberate sweep over
small rewind depths, which this experiment did not run.

### Controls

- **Identical episodes.** Seeds, faults, supervisor draws and hyperparameters are
  the same across conditions. Only the rewind target changes.
- **Matched dataset size.** Rewinding further changes how many corrective frames
  are produced — and not in the obvious direction: `onset` yields *more*, because
  recovering from a failure takes longer than doing the task correctly. Every
  dataset is subsampled to the size of the smallest, so "corrected the right
  states" is separated from "had more data". The subsample is drawn uniformly
  across each source's frames, so all 188 corrective episodes still contribute
  under every condition; capping by truncation would have dropped whole episodes
  and reintroduced a coverage difference through the back door.
- **Matched initial-state coverage.** In the controlled variant, one pool of 150
  scripted demonstrations is collected once and reused byte-for-byte by all four
  conditions. This is the control that changed the answer, and it is checked by a
  regression test: an earlier version silently dropped the pool, and the
  "controlled" run reproduced the uncontrolled one to three decimal places.
- **Confidence intervals on everything.** Wilson intervals, and the comparison
  says explicitly when two conditions are not distinguishable. At n=50, 80% vs
  60% is *not* a resolved difference, and reporting it as one would be wrong.

---

## What this design cannot tell you

- **Nothing here is a human result.** The supervisor is simulated and
  `trace_accuracy` is a free parameter. The deliverable is the curve across it,
  so that a VR study with participants can be read off it later.
- **The agent under supervision is a scripted expert with an injected fault**,
  not a learned policy. Its failure mode is known exactly, which is the point,
  but real policies fail in messier and more correlated ways.
- **Simulation only.** No sim-to-real claim is made or supported.
- **One task family.** Four phases, one object type, one goal. Whether the effect
  scales with task length is untested.

## What the results predict

Two things, from the two experiments.

**From the attribution result.** Misattribution is driven by *observability*
rather than causal distance — `wrong_object` is attributed perfectly despite
having the largest causal gap in the set, because picking the wrong block is
visible immediately. If that holds for people, the cheapest intervention is not
asking them to reason harder about causes. It is surfacing failures earlier: a
grip-force readout, a slip indicator, anything that moves the symptom closer to
the cause. That is what the VR study should test first.

**From the degradation result.** Attribute for analysis; correct at the symptom.
Knowing that a lift-phase drop was really a grasp-phase error is worth a great
deal for diagnosis — it tells you what to fix in the policy, the gripper or the
task design. It is worth negative value as a *rewind target*, because acting on
it moves the correction off the states the policy actually fails in. The two uses
of an attribution are separable, and this experiment says only the first pays.

The corollary for anyone building such a system: measure where your corrections
land before you attribute a difference to their quality. Both offline proxies
available here — how often the corrections themselves succeed, and held-out loss
— ranked the strategies in exactly the wrong order, because the protocol changed
which states were in the dataset and the held-out set moved with it.
