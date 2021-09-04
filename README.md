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
        # From "" => prob of 0: 0.4, 1: 0.6
        return [log(.4), log(.6)]
    if len(sequence) == 1 and sequence[0] == 0:
        # From "0" => prob of 0: 0.5, 1: 0.5
        return [log(.5), log(.5)]
    # From "1", "00", "01" => terminal
    return None # indicates that after this sequence prefix there should be no further sampling

max_categories: int = 2 # since at any decision there is at most 2 choices
seed: int = 0
# Create a sequence generator, it can be used until you exhaust it i.e. you sampled everything
gen: SequenceGenerator = SequenceGenerator(lambda sequences: [get_logprobs(seq) for seq in sequences], max_categories, seed)
# We sample 2 sequences
sequence_list: list[list[int]] = sample(gen, 2) 
# We can sample again a batch of 2
sequence_list_two: list[list[int]] = sample(gen, 2) 
# However since there are only three sequences it only returns one: the missing sequence
assert len(sequence_list_two) == 1
```

## Installation

### For production

```bash
pip install cython # must be installed first
pip install git+https://github.com/Theomat/sbsur
```

### For development

```bash
pip install cython # must be installed first
pip intall numpy pytest # install development dependencies
git clone https://github.com/Theomat/sbsur
pip install -e ./sbsur
```

Then you can run the tests using the command: `pytest`.

## Performance Comparison

Here we show the performance gain of using our implementation compared to the [Unique Randomizer](https://github.com/google-research/unique-randomizer)(UR) implementation of the combination.
The performance comparison is done on a single thread on a i7-9750H.
We test the performance on different configurations of sequences with `k` categories of length `n` sampled with a batch size `b`.
We measure the average time it takes to sample in a given configuration until exhaustion of the sequences.

### Speed

| k   | n | b      | Samples   | Time/Sample | SBSUR (us) | [UR](https://github.com/google-research/unique-randomizer) |
|-----|---|--------|-----------|-------------|------------|------|
| 10  | 5 | 10     | 100,000   | 10.3µs      | x1         | x31  |
| 10  | 5 | 100    | 100,000   | 9.2 µs      | x1         | x28  |
| 10  | 6 | 100    | 1,000,000 | 5.72µs      | x1         | x59  |
| 20  | 5 | 100    | 3,200,000 | 7.06µs      | x1         | x37  |
| 200 | 3 | 1,000  | 8,000,000 | 10.2µs      | x1         | x16  |
| 200 | 3 | 10,000 | 8,000,000 | 10.1µs      | x1         | x16  |

### Memory

| k   | n | b      | Samples   | Max Memory | SBSUR (us) | [UR](https://github.com/google-research/unique-randomizer) |
|-----|---|--------|-----------|------------|------------|--------|
| 10  | 5 | 10     | 100,000   | 40MB       | x1         | x2.5   |
| 10  | 5 | 100    | 100,000   | 40MB       | x1         | x2.5   |
| 10  | 6 | 100    | 1,000,000 | 86MB       | x1         | x7.3   |
| 20  | 5 | 100    | 3,200,000 | 181MB      | x1         | x9.8   |
| 200 | 3 | 1,000  | 8,000,000 | 339MB      | x1         | x11.3  |
| 200 | 3 | 10,000 | 8,000,000 | 279MB      | x1         | x14.0  |

### Reproduction

To reproduce the results of the table above, you will need to have installed both SBSUR and [Unique Randomizer](https://github.com/google-research/unique-randomizer).
Once installed you can run:

```bash
./experiments/run_experiment.sh k n b
```

## Multithreading

In Python, multithreading is organised around a Global Interpreter Lock (GIL). In the case of CPU bound tasks such as ours, the GIL is never released. However in Cython one can use the `nogil` qualifier to express that the GIL is not required, this qualifier add constraints on the development. We did not use the `nogil` qualifier in this project, to use this qualifier, the code should be ported to C++ since it entails that Python objects should not be modified. Note that the code can't be made completely GIL-free since we use a python callback.

 Instead you can use `multiprocessing`, each process will have its own GIL so you will benefit from performances improvements. Furthermore, except for the GIL, the code has been designed to not use global locks.

## Possible improvements

- (+ Speed - Memory) Move everything to C++.

- (- Speed - Memory) Since it is often the case the in `ur_get_logprobs` the log probs are re computed, instead directly compute them in a provided buffer, thus removing the need for the `double* logprobs` array in `ur_node_t`. Though this might cost some speed.