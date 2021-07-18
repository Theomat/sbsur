# distutils: language=c++
from libcpp.vector cimport vector
from random_wrapper cimport mt19937_64
from unique_randomizer cimport ur_node_t


cdef class SequenceGenerator:
    # No underscore since the attributes are private by default

    # get_log_probs is a Python callback
    #   get_log_probs: vector<int> -> double* is the C++ signature
    #   get_log_probs: list[int] -> Optional[Union[list[float], np.ndarray[float]]] is  the Python signature
    cdef object pyfun_get_log_probs
    cdef ur_node_t* root
    cdef int max_categories
    cdef mt19937_64 generator

    cdef double* get_log_probs(self, vector[int] sequence_prefix, int* categories_ptr)
    cdef ur_node_t* get_state(self)
    cdef int get_max_categories(self)
    cdef mt19937_64* get_generator(self)
