# sbsur

[Stochastic Beam Search](https://github.com/wouterkool/stochastic-beam-search) + [Unique Randomizer](https://github.com/google-research/unique-randomizer)

The original implementation in the two linked repositories are implemented in pure Python with some numpy support.
We kept the original ease of use while implementing it in [Cython](https://cython.org/) to get a speed increase.
This project defines a Python interface and a Cython interface, with no dependencies but [Cython](https://cython.org/).

## Example

The following example simply generates the following sequences: `1`, `00`, `01`.

```python
from math import log
from sbsur import SequenceGenerator, sample

def get_logprobs(sequence: list[int]) -> Optional[list[float]]:
    if len(sequence) == 0:
        return [log(.4), log(.6)]
    if len(sequence) == 1 and sequence[0] == 0:
        return [log(.5), log(.5)]
    return None # indicates that after this sequence prefix there should be no further sampling

max_categories: int = 2 # since at any decision there is at most 2 choices
seed: int = 0
# Create a sequence generator, it can used until you exhaust it i.e. you sampled everything
gen: SequenceGenerator = SequenceGenerator(get_logprobs, max_categories, seed)
# We sample 2 sequences
sequence_list: list[list[int]] = sample(gen, 2) 
# We can sample again a batch of 2
sequence_list_two: list[list[int]] = sample(gen, 2) 
# However since there are only three sequences it only returns one: the missing sequence
assert len(sequence_list_two) == 1
```

## Installation

You can skip the command to install dependencies, they should be installed automatically when installing `sbsur`.

### For production

```bash
pip install cython # install production dependencies
pip install git+https://github.com/Theomat/sbsur
```

### For development

```bash
pip install cython numpy pytest # install development dependencies
git clone https://github.com/Theomat/sbsur
pip install -e ./sbsur
```

Then you can run the tests using the command: `pytest`.

## Performance Comparison

Here we show the performance gain of using our implementation compared to the [Unique Randomizer](https://github.com/google-research/unique-randomizer)(UR) implementation of the combination.
The performance comparison is done on a single thread on a i7-9750H.
We test the performance on different configurations of sequences with `k` categories of length `n` sampled with a batch size `b`.
We measure the average time it takes to sample in a given configuration until exhaustion of the sequences.

| Use Case            | Samples    | Time/Sample | SBSUR (us) | [UR](https://github.com/google-research/unique-randomizer) |
|---------------------|-----------|-------------|------------|----------|
| `k=10, n=5, b=10`   | 100,000   | 5.670µs     | x1         | x23.0820 |
| `k=10, n=5, b=100`  | 100,000   | 4.285µs     | x1         | x30.5947 |
| `k=10, n=6, b=10`   | 1,000,000 | 6.579µs     | x1         | x20.6190
| `k=20, n=5, b=10`   | 3,200,000 | 8.753µs     | x1         | x17.7002
| `k=200, n=3, b=200` | 8,000,000 |

### Reproduction

To reproduce the results of the table above, you will need to have installed both SBSUR and [Unique Randomizer](https://github.com/google-research/unique-randomizer).
Once installed you can run:

```bash
python experiments/speed_experiment.py
```

## Multithreading

In Python, multithreading is organised around a Global Interpreter Lock (GIL). In the case of CPU bound tasks such as ours, the GIL is never released. However in Cython one can use the `nogil` qualifier to express that the GIL is not required, this qualifier add constraints on the development. We did not use the `nogil` qualifier in this project, to use this qualifier, the code should be ported to C++ since it entails that Python objects should not be modified. Note that the code can't be made completely GIL-free since we use apython callback.

 Instead you can use `multiprocessing`, each process will have its own GIL so you will benefit from performances improvements. Furthermore, except for the GIL, the code has been designed to not use global locks.
