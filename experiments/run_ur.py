from typing import Tuple
import numpy as np
import torch
from unique_randomizer import unique_randomizer as ur
import warnings
# for ur.SBS
warnings.filterwarnings('ignore')
def makes_ur_functions(probs: np.ndarray):
    def child_log_probability_fn(states: list[list[int]]) -> list[np.ndarray]:
        results = []
        for state in states:
            for i, el in enumerate(state):
              if el is None:
                results.append(np.log(probs[:, i]))
                break
        return results
    def child_state_fn(state_index_pairs: list[Tuple[list[int], int]]) -> list[Tuple[list[int], bool]]:
        """Produces child states."""
        results = []
        for state, index in state_index_pairs:
          j = -1
          for i, el in enumerate(state):
              if el is None:
                j = i
                break
          if j >= 0:
            new_state = [x for x in state]
            new_state[j] = index
            if j + 1 == probs.shape[1]:
              results.append((new_state, False))
            else:
              results.append((new_state, True))
        return results
    return child_log_probability_fn, child_state_fn

def do_sampling(n: int, k: int, b: int):
    np.random.seed(0)
    p = np.random.uniform(0, 1, size=k * n).reshape((k, n))
    for i in range(n):
      p[:, i] /= np.sum(p[:, i])


    samples : int = int(k**n)
    iterations = samples // b
    assert samples % b == 0

    randomizer = ur.UniqueRandomizer()
    child_log_probability_fn, child_state_fn = makes_ur_functions(p)
    for _ in range(iterations):
        randomizer.sample_batch(
          child_log_probability_fn=child_log_probability_fn,
          child_state_fn=child_state_fn,
          root_state=[None for _ in range(n)],
          k=b)



if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("There must be 3 arguments: k, n, b")
        sys.exit(1)
    k, n, b = [int(x) for x in sys.argv[1:]]
    do_sampling(n, k, b)