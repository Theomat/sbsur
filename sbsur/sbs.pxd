# distutils: language=c++
from libcpp.vector cimport vector

from sequence_generator cimport SequenceGenerator

# We hand out a vector (instead of an array) so that it's automatically freed
cdef vector[vector[int]] c_sample(SequenceGenerator generator, int batch_size)