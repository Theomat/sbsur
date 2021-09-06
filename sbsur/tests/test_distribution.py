from typing import Optional
from sbsur import sample, SequenceGenerator

import collections
from math import log
import numpy as np
import pytest


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
    results = []
    for i in range(10000):
      seq_gen = SequenceGenerator(lambda x: [get_logprobs(l) for l in x], 2, i)
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


# Test parameters
# Normal approximation doesn't work well when probabilities are low
# So use low depth
DEPTH: int = 3
CATEGORIES: int = 5
TESTS: int = 10
SAMPLES: int = int(1e5)
np.random.seed(0)
testdata = [np.random.uniform(size=CATEGORIES*DEPTH) for _ in range(TESTS)]


def make_logprobs_getter(probs: np.ndarray):
    def get_logprobs(sequence: list[int]) -> Optional[list[float]]:
        if len(sequence) < len(probs):
            return np.log(probs[len(sequence)])
        return None
    return lambda x: [get_logprobs(el) for el in x]

@pytest.mark.parametrize("probabilities", testdata)
def test_uniform_trees(probabilities: np.ndarray):
    probabilities = probabilities.reshape((DEPTH, -1))
    for i in range(DEPTH):
        probabilities[i, :] /= np.sum(probabilities[i, :])


    results = []
    for i in range(SAMPLES):
        gen = SequenceGenerator(make_logprobs_getter(probabilities), CATEGORIES, i)
        sequences_list = sample(gen, 1)
        results.append(tuple(sequences_list[0]))
    counter = collections.Counter(results)
    ci = 0.95
    alpha = 1 - ci
    z = 2 * (1 - alpha / 2)
    total = len(counter)
    failed = 0
    for sequence, count in counter.items():
        p = np.prod([probabilities[i, j] for i, j in enumerate(sequence)])
        # count ~ Binomial(p, SAMPLES)
        p_hat = count / SAMPLES
        # 95% confidence interval
        atol = z * np.sqrt(p_hat * (1 - p_hat) / SAMPLES) * SAMPLES
        expected_count =  p * SAMPLES
        failed += not np.isclose(count, expected_count, rtol=0, atol=atol)

    # failed ~ Binomial(0.05, total)
    p_hat = failed / total
    max_failed = .05 * total + z * np.sqrt(p_hat * (1 - p_hat) / total) * total
    assert failed <= max_failed