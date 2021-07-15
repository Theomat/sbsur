# distutils: language=c++
from libcpp.vector cimport vector

from unique_randomizer cimport ur_node_t


cdef class SequenceGenerator:
    cdef double* get_log_probs(vector[int] sequence_prefix)
    cdef ur_node_t* get_state()
