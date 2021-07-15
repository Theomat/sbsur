# distutils: language=c++
from libcpp.vector cimport vector

from unique_randomizer cimport ur_node_t


cdef class SequenceGenerator:
    # No underscore since the attributes are private by default

    # get_log_probs is a Python callback
    #   get_log_probs: vector<int> -> double* is the C++ signature
    #   get_log_probs: list[int] -> Optional[Union[list[float], np.ndarray[float]]] is  the Python signature
    cdef void* pyfun_get_log_probs
    cdef ur_node_t* root

    cdef double* get_log_probs(self, vector[int] sequence_prefix)
    cdef ur_node_t* get_state(self)
