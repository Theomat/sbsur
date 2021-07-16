# distutils: language=c++
from libcpp.vector cimport vector

from sequence_generator cimport SequenceGenerator

# We hand out a vector (instead of an array) of sequences so that it's automatically freed
# Note that the sequences that we output are in reversed order
cdef vector[(vector[int], float)] c_sample(SequenceGenerator generator, int batch_size)