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
testdata = [np.random.uniform(size=CATEGORIES*DEPTH) for _ in range(TESTS)]


def make_logprobs_getter(probs: np.ndarray):
    def get_logprobs(sequence: list[int]) -> Optional[list[float]]:
        if len(sequence) < probs.shape[0]:
        return np.log(probs[:, len(sequence)])
        return None
    return get_logprobs

@pytest.mark.parametrize("probabilities", testdata)
def test_enumeration(probabilities: np.ndarray):
    probabilities = probabilities.reshape((-1, DEPTH))
    for i in range(DEPTH):
        probabilities[:, i] /= np.sum(probabilities[:, i])

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
    