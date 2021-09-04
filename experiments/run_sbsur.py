from typing import Optional
import numpy as np
from sbsur import SequenceGenerator, sample

def make_logprobs_getter(probs: np.ndarray):
    def get_logprobs(sequence: list[int]) -> Optional[list[float]]:
        if len(sequence) < probs.shape[1]:
            return np.log(probs[:, len(sequence)])
        return None
    return lambda x: [get_logprobs(el) for el in x]


def do_sampling(n: int, k: int, b: int):
    np.random.seed(0)
    p = np.random.uniform(0, 1, size=k * n).reshape((k, n))
    for i in range(n):
      p[:, i] /= np.sum(p[:, i])


    get_logprobs = make_logprobs_getter(p)

    samples : int = int(k**n)
    iterations = samples // b
    assert samples % b == 0

    r = SequenceGenerator(get_logprobs, k, 0)
    for _ in range(iterations):
        sample(r, b)



if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("There must be 3 arguments: k, n, b")
        sys.exit(1)
    k, n, b = [int(x) for x in sys.argv[1:]]
    do_sampling(n, k, b)