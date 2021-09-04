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
    li = []
    def get_logprobs(sequence: list[int]) -> Optional[list[float]]:
        li.append(sequence)
        if len(sequence) < len(probs):
            return np.log(probs[len(sequence)])
        return None
    return lambda x: [get_logprobs(el) for el in x], li

@pytest.mark.parametrize("probabilities", testdata)
def test_uniform_trees(probabilities: np.ndarray):
    probabilities = probabilities.reshape((DEPTH, -1))
    for i in range(DEPTH):
        probabilities[i, :] /= np.sum(probabilities[i, :])

    get_logprobs, l = make_logprobs_getter(probabilities)
    gen = SequenceGenerator(get_logprobs, CATEGORIES, 0)
    for _ in range(SAMPLES // BATCH_SIZE):
        sample(gen, BATCH_SIZE)
        for seq in l:
            for i, x in enumerate(seq):
                if not( x >= 0 and x < probabilities.shape[1]):
                    print("Asked for the probabilitfy of invalid sequence:",seq, "allowed:", probabilities.shape[1])
                    assert False
        l.clear()


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
    get_logprobs, l = make_logprobs_getter(probabilities)
    gen = SequenceGenerator(get_logprobs, CATEGORIES, 0)
    iterations = int(np.ceil(samples / BATCH_SIZE))
    for _ in range(iterations):
        sample(gen, BATCH_SIZE)
        for seq in l:
            for i, x in enumerate(seq):
                if not( x >= 0 and x < k[i]):
                    print("Asked for the probabilitfy of invalid sequence:",seq, "allowed:", k)
                    assert False
        l.clear()
