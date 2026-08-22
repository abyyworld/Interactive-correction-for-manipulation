import pytest

from interventionkit.schema import (
    SCHEMA_VERSION,
    EpisodeMeta,
    InterventionSegment,
    segments_from_actors,
)


def test_segments_from_actors_finds_contiguous_runs():
    actors = ["policy"] * 3 + ["human"] * 4 + ["policy"] * 2 + ["human"] * 2
    phases = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3]
    segs = segments_from_actors(actors, phases)
    assert [(s.start, s.end) for s in segs] == [(3, 7), (9, 11)]
    assert segs[0].onset_phase == 1  # phase at the moment of takeover
    assert segs[0].duration == 4


def test_segments_handles_trailing_and_empty():
    assert segments_from_actors(["policy"] * 5) == []
    assert segments_from_actors(["human"] * 3)[0].end == 3
    assert segments_from_actors([]) == []


def test_expert_counts_as_intervention():
    segs = segments_from_actors(["policy", "expert", "expert", "policy"])
    assert len(segs) == 1 and segs[0].start == 1 and segs[0].end == 3


def test_episode_meta_roundtrip():
    seg = InterventionSegment(start=1, end=4, onset_phase=2, attributed_phase=0, confidence=0.5)
    meta = EpisodeMeta(
        episode_id="e", task="t", seed=7, n_steps=10, success=False, interventions=[seg]
    )
    back = EpisodeMeta.from_dict(meta.to_dict())
    assert back.to_json() == meta.to_json()
    assert back.interventions[0].attributed_phase == 0
    assert back.n_corrected_steps == 3
    assert back.intervened


def test_future_schema_version_is_rejected():
    """A dataset that loads but means something different is worse than one that fails."""
    d = EpisodeMeta(episode_id="e", task="t", seed=0, n_steps=1, success=True).to_dict()
    d["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema version"):
        EpisodeMeta.from_dict(d)
