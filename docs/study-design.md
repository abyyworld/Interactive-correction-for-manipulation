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

### Why the corrections differ at all

With `oracle`, the corrective demonstration begins from the *pre-failure* state —
the states the policy actually visits when it goes wrong — so it learns what to
do there. With `onset`, it begins from the *post-failure* state, where the object
has already been dropped or knocked away. Those states only occur after the
policy has already failed, so training on them teaches recovery from a situation
the policy will keep entering.

### Controls

- **Identical episodes.** Seeds, faults, supervisor draws and hyperparameters are
  the same across conditions. Only the rewind target changes.
- **Matched dataset size.** Rewinding further changes how many corrective frames
  are produced — and not in the obvious direction: `onset` yields *more*, because
  recovering from a failure takes longer than doing the task correctly. Every
  dataset is subsampled to the size of the smallest, so "corrected the right
  states" is separated from "had more data".
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

## What the result predicts

If misattribution is driven by *observability* rather than causal distance — as
`wrong_object` suggests, being attributed perfectly despite the largest causal
gap — then the cheapest intervention is not asking people to reason harder about
causes. It is surfacing failures earlier: a grip-force readout, a slip indicator,
anything that moves the symptom closer to the cause.

That is a testable prediction, and it is the one the VR study should test first.
