from typing import Optional
from sbsur import sample, SequenceGenerator

import numpy as np
import pytest

# Test parameters
DEPTH: int = 5
CATEGORIES: int = 5
TESTS: int = 10
# Constants for test
BATCH_SIZE : int = CATEGORIES
SIZE: int = DEPTH * 5
SAMPLES: int = int(CATEGORIES**DEPTH)
np.random.seed(0)
testdata = [np.random.uniform(size=CATEGORIES*DEPTH) for _ in range(TESTS)]


def make_logprobs_getter(probs: np.ndarray):
    def get_logprobs(sequence: list[int]) -> Optional[list[float]]:
        if len(sequence) < len(probs):
            return np.log(probs[len(sequence)])
        return None
    return get_logprobs

@pytest.mark.parametrize("probabilities", testdata)
def test_uniform_trees(probabilities: np.ndarray):
    probabilities = probabilities.reshape((DEPTH, -1))
    for i in range(DEPTH):
        probabilities[i, :] /= np.sum(probabilities[i, :])

    gen = SequenceGenerator(make_logprobs_getter(probabilities), CATEGORIES, 0)
    results = []
    for _ in range(SAMPLES // BATCH_SIZE):
        sequences_list = sample(gen, BATCH_SIZE)
        for s in sequences_list:
            sequence = "".join([str(x) for x in s])
            if sequence in results:
                print("failed to sample unique sequence:", np.unique(results).shape[0], "/", SAMPLES)
                assert False
            results.append(sequence)
    assert len(results) == SAMPLES
    assert gen.is_exhausted()
    assert all(len(s) == DEPTH for s in results)
    assert all(all(int(x) >= 0 and int(x) < CATEGORIES for x in s) for s in results)



test_seeds = [i for i in range(TESTS)]

@pytest.mark.parametrize("seed", test_seeds)
def test_non_uniform_tree(seed: int):
    probabilities = []
    np.random.seed(seed)
    k = np.random.randint(2, CATEGORIES + 1, size=DEPTH)
    for i in range(DEPTH):
        p = np.random.uniform(size=k[i])
        p /= np.sum(p)
        probabilities.append(p)

    samples = np.prod(k)
    gen = SequenceGenerator(make_logprobs_getter(probabilities), CATEGORIES, 0)
    results = []
    iterations = int(np.ceil(samples / BATCH_SIZE))
    for _ in range(iterations):
        sequences_list = sample(gen, BATCH_SIZE)
        for s in sequences_list:
            sequence = "".join([str(x) for x in s])
            if sequence in results:
                print("failed to sample unique sequence:", np.unique(results).shape[0], "/", SAMPLES)
                assert False
            results.append(sequence)
    assert len(results) == samples
    assert gen.is_exhausted()
    assert all(len(s) == DEPTH for s in results)
    assert all(all(int(x) >= 0 and int(x) < k[i] for i, x in enumerate(s)) for s in results)