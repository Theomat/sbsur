
from typing import Optional, Tuple
import chronometer
import numpy as np

from sbsur import SequenceGenerator, sample

from unique_randomizer import unique_randomizer as ur

import warnings
# for ur.SBS
warnings.filterwarnings('ignore')

def print_line(s1, s2, s3, s4, s5):
  line = "{:<28} {:<14} {:<11} {:<8} {:<8}"
  print(line.format(s1, s2, s3, s4, s5))

def make_logprobs_getter(probs: np.ndarray):
    def get_logprobs(sequence: list[int]) -> Optional[list[float]]:
        if len(sequence) < probs.shape[0]:
            return np.log(probs[:, len(sequence)])
        return None
    return get_logprobs


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


def test_speed(n: int, k: int, b: int):
    np.random.seed(0)
    p = np.random.uniform(0, 1, size=k * n).reshape((k, n))
    for i in range(n):
      p[:, i] /= np.sum(p[:, i])

    chronometer.reset("ur")
    chronometer.reset("sbsur")

    get_logprobs = make_logprobs_getter(p)

    samples : int = int(k**n)
    iterations = samples // b
    assert samples % b == 0

    r = SequenceGenerator(get_logprobs, k, 0)
    for _ in range(iterations):
      with chronometer.clock("sbsur"):
        sample(r, b)

    avg = chronometer.get("sbsur").total / samples
    
    randomizer = ur.UniqueRandomizer()
    child_log_probability_fn, child_state_fn = makes_ur_functions(p)
    for _ in range(samples):
      with chronometer.clock("ur"):
        randomizer.sample_batch(
          child_log_probability_fn=child_log_probability_fn,
          child_state_fn=child_state_fn,
          root_state=[None for _ in range(n)],
          k=b)
    ratio = chronometer.get("ur").total / chronometer.get("sbsur").total
    print_line(f"k={k}, n={n}, b={b}", f"{samples}", f"{avg* 1e6:.3f}µs", "x1", f"x{ratio:.4f}")


if __name__ == "__main__":
    configurations = [
      {"k": 10, "n": 5, "b": 10},
      {"k": 10, "n": 5, "b": 100},
      {"k": 10, "n": 6, "b": 10},
      {"k": 20, "n": 5, "b": 1000},
      {"k": 200, "n": 3, "b": 8000},
    ]
    print_line("config", "samples", "time/sample", "SBSUR", "UR")
    for config in configurations:
      test_speed(**config)