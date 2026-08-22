import numpy as np

from tests.conftest import requires_assets

pytestmark = requires_assets


def test_supervisor_detects_and_rescues_failures(env):
    from icm.envs.faults import FaultInjector, FaultSpec, FaultType
    from icm.eval.rollout import ScriptedAgent, rollout
    from icm.study.supervisor import SupervisorConfig, SyntheticSupervisor

    supervisor = SyntheticSupervisor(SupervisorConfig(), rng=np.random.default_rng(0))
    detected = rescued = 0
    n = 12
    for i in range(n):
        injector = FaultInjector(
            FaultSpec(type=FaultType.WEAK_GRIP, severity=0.9), np.random.default_rng(i)
        )
        result = rollout(
            env, ScriptedAgent(injector=injector), supervisor=supervisor, seed=9500 + i
        )
        detected += int(result.intervened)
        rescued += int(result.success)
    assert detected == n
    assert rescued >= n * 0.6, f"corrections only rescued {rescued}/{n}"


def test_takeover_does_not_drop_a_held_object(env):
    """Resetting the corrective expert opens the gripper, manufacturing a failure."""
    from icm.control.scripted import ScriptedExpert, Stage

    env.reset(seed=5)
    expert = ScriptedExpert()
    expert.reset()
    for _ in range(env.config.max_episode_steps):
        env.step(expert.act(env))
        if env.is_grasped() and env.object_pos()[2] > 0.06:
            break
    assert env.is_grasped()

    taking_over = ScriptedExpert()
    taking_over.resume_from_state(env)
    assert taking_over.stage in (Stage.LIFT, Stage.TRANSPORT)
    assert taking_over.gripper_close_value == -1.0  # keeps holding


def test_supervisor_rarely_intervenes_on_healthy_episodes(env):
    """False positives record an intervention with no root cause."""
    from icm.eval.rollout import ScriptedAgent, rollout
    from icm.study.supervisor import SupervisorConfig, SyntheticSupervisor

    supervisor = SyntheticSupervisor(SupervisorConfig(), rng=np.random.default_rng(0))
    n = 20
    false_positives = sum(
        rollout(env, ScriptedAgent(), supervisor=supervisor, seed=9000 + i).intervened
        for i in range(n)
    )
    assert false_positives <= n * 0.30, f"{false_positives}/{n} unnecessary interventions"


def test_trace_accuracy_controls_stated_attribution(env):
    """The swept parameter must actually move the reported cause."""
    from icm.envs.phases import Phase
    from icm.study.supervisor import SupervisorConfig, SyntheticSupervisor

    for accuracy, expected in ((0.0, False), (1.0, True)):
        supervisor = SyntheticSupervisor(
            SupervisorConfig(trace_accuracy=accuracy), rng=np.random.default_rng(0)
        )
        supervisor.true_root_phase = Phase.APPROACH
        reports = [supervisor._attribute(Phase.LIFT) for _ in range(20)]
        matched = all(r == Phase.APPROACH for r in reports)
        assert matched is expected


def test_detection_never_reads_the_injected_fault():
    """Detection must depend only on observable state.

    Checked structurally rather than behaviourally: if the detector ever gained
    access to the ground-truth fault, every detection-lag and misattribution
    number in the study would be meaningless, and that is not the kind of
    regression a sampling-based test reliably catches.
    """
    import ast
    import inspect
    import textwrap

    from icm.study.supervisor import SyntheticSupervisor

    tree = ast.parse(textwrap.dedent(inspect.getsource(SyntheticSupervisor._detect)))
    # Compare against identifiers only, so prose in the docstring cannot trip it.
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    # "spec" is deliberately absent: _detect iterates env.object_specs, which is
    # scene description, not privileged fault information.
    forbidden = {
        "injector",
        "fault",
        "true_root_phase",
        "ground_truth",
        "root_phase",
        "root_onset_step",
        "symptom_phase",
    }
    leaked = names & forbidden
    assert not leaked, f"_detect references privileged state: {leaked}"
