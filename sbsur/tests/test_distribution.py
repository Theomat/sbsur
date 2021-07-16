from typing import Optional
from sbsur import sample, SequenceGenerator

import collections
from math import log



def get_logprobs(sequence: list[int]) -> Optional[list[float]]:
    if len(sequence) == 0:
        return [log(.4), log(.6)]
    if len(sequence) == 1 and sequence[0] == 0:
        return [log(.5), log(.5)]
    return None


def seq2output(sequences: list[int]) -> str:
    out: str = ""
    for seq in sequences:
        if seq[0] == 1:
            out += "2"
        else:
            out += str(seq[1])
    return out

def test_proportions():
    # This test is analogous to unique_randomizer's test_proportions.
    # It's possible but extremely unlikely for this test to fail.

    # A state is a pair representing two coin flips. The first flip is biased
    # (60% True). If True, the output is '2'. If False, then the second flip
    # (fair odds) determines whether the output is '0' or '1'.
    # See unique_randomizer_test.py's test_proportions for a procedural
    # representation of this logic.

    results = []
    for _ in range(10000):
      seq_gen = SequenceGenerator(get_logprobs, 2)
      seqs = sample(seq_gen, 3)
      results.append(seq2output(seqs))

    assert all(len(s) == 3 for s in results)
    counter = collections.Counter(results)

    # P('201') = P('210') = 0.6 * 0.5.
    assert abs(counter['201'] - 0.6 * 0.5 * 10000) <= 250
    assert abs(counter['210'] - 0.6 * 0.5 * 10000) <= 250

    # P('021') = P('120') = 0.2 * 0.75.
    assert abs(counter['021'] - 0.2 * 0.75 * 10000) <= 200
    assert abs(counter['120'] - 0.2 * 0.75 * 10000) <= 200

    # P('012') = P('102') = 0.2 * 0.25.
    assert abs(counter['012'] - 0.2 * 0.25 * 10000) <= 100
    assert abs(counter['102'] - 0.2 * 0.25 * 10000) <= 100