"""Templated language conditioning.

Scope, stated plainly
---------------------
Instructions are generated from a template ("pick up the {colour} {shape} and
place it on the pad") and encoded as a bag of words over a closed
vocabulary. This is **not** free-form natural language and should not be
described as such. It is the smallest thing that makes the task genuinely
language-conditioned: the policy cannot succeed without reading the instruction,
because which object is the target changes between episodes.

That last point is the part that is easy to get wrong. If the target is always
the red block, a "language-conditioned" policy can ignore the text completely
and still score well, and the benchmark measures nothing. ``EnvConfig
.randomize_target`` exists to close that hole, and
:func:`instruction_is_load_bearing` asserts it.

A sentence-embedding backend (CLIP, sentence-transformers) can be swapped in
behind the same interface when a network is available; bag-of-words keeps the
default installable and deterministic offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

TOKEN_RE = re.compile(r"[a-z]+")

#: Words that carry no discriminative signal in these templates. Dropping them
#: keeps the vocabulary small enough that a bag of words is not mostly noise.
STOPWORDS = frozenset({"the", "a", "an", "and", "on", "it", "to", "put", "place", "up", "pick"})


def tokenize(text: str) -> list[str]:
    return [w for w in TOKEN_RE.findall(text.lower()) if w not in STOPWORDS]


@dataclass
class Vocabulary:
    """Closed vocabulary built from the instructions actually present in a dataset."""

    words: tuple[str, ...]

    def __post_init__(self) -> None:
        self._index = {w: i for i, w in enumerate(self.words)}

    def __len__(self) -> int:
        return len(self.words)

    def encode(self, instruction: str) -> np.ndarray:
        """Multi-hot bag of words. Unknown words are ignored, not an error.

        Ignoring rather than raising matters at deployment: an operator typing a
        synonym should degrade the instruction, not crash the robot.
        """
        vec = np.zeros(len(self.words), dtype=np.float32)
        for word in tokenize(instruction):
            idx = self._index.get(word)
            if idx is not None:
                vec[idx] = 1.0
        return vec

    @classmethod
    def from_instructions(cls, instructions) -> Vocabulary:
        words = sorted({w for text in instructions for w in tokenize(text)})
        return cls(tuple(words))

    @classmethod
    def from_object_specs(cls, specs) -> Vocabulary:
        words = set()
        for spec in specs:
            words.update(tokenize(f"{spec.color_word} {spec.shape_word}"))
        words.add("pad")
        return cls(tuple(sorted(words)))


def build_vocabulary_from_run(run_dir) -> Vocabulary:
    """Read a recorded run's instructions and build the vocabulary from them."""
    from interventionkit import RunReader

    reader = RunReader(run_dir)
    return Vocabulary.from_instructions(m.instruction for m in reader.episodes() if m.instruction)


def instruction_is_load_bearing(instructions, targets) -> bool:
    """True when the instruction actually determines the target.

    Guards the failure mode where every episode shares one instruction, so a
    policy that ignores language scores identically to one that reads it.
    """
    unique_instructions = {str(i) for i in instructions}
    unique_targets = {str(t) for t in targets}
    return len(unique_instructions) > 1 and len(unique_targets) > 1
