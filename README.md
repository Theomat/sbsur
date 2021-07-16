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
    if len(sequences) == 0:
        return [log(.4), log(.6)]
    if len(sequences) == 1 and sequence[0] == 0:
        return [log(.5), log(.5)]
    return None

max_categories: int = 2 # since at any decision there is at most 2 choices
seed: int = 0
# Create a sequence generator, it can used until you exhaust it i.e. you sampled everything
gen: SequenceGenerator = SequenceGenerator(get_logprobs, max_categories, seed)
# We sample 2 sequences
sequence_list: list[list[int]] = sample(gen, 2) 
# We can sample again a batch of 2
sequence_list_two: list[list[int]] = sample(gen, 2) 
# However since there are only three sequences it only returns ones: the missign sequence
assert len(sequence_list_two) == 1
```

## Installation

For a production usage:

```bash
pip install cython
pip install git+https://github.com/Theomat/sbsur
```

For development:

```bash
pip install cython numpy pytest
pip install git+https://github.com/Theomat/sbsur
```

## Multithreading

In Python, multithreading is organised around a Global Interpreter Lock (GIL). In the case of CPU bound tasks such as ours, the GIL is never released. However in Cython one can use the nogil qualifier to express that the GIL is not required, this qualifier add constraints on the development of Cython and this was not done in this project, to use this qualifier, the code should be ported to C++. Instead you can use multiprocessing, each process will have its own GIL so you will benefit from performances improvements. Furthermore, except for the GIL, the code has been designed to not use global locks.
