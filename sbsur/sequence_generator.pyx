# distutils: language=c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
from libcpp.vector cimport vector
from libcpp cimport bool
# Use the cython ones, they are thread-safe and give stats to python memory manager while behaving like C-ones (no GIL)
from cpython.mem cimport PyMem_Malloc
from random_wrapper cimport mt19937_64, random_device

from unique_randomizer cimport ur_node_t, ur_free_all, ur_new, ur_set_logprobs, ur_is_exhausted


cdef class SequenceGenerator:

    def __cinit__(self, python_callback, int max_categories, seed):
        self.pyfun_get_log_probs = python_callback
        if not callable(python_callback):
            raise TypeError("The first argument must be a callable!")
        self.root = ur_new()
        cdef int categories = 0
        cdef vector[int] empty = vector[int]()
        cdef double* logprobs = self.get_log_probs(empty, &categories)
        ur_set_logprobs(self.root, logprobs, categories)
        self.max_categories = max_categories
        cdef random_device r
        if seed is None:
            self.generator = mt19937_64(r())
        else:
            self.generator = mt19937_64(seed)

    cdef double* get_log_probs(self, vector[int] sequence_prefix, int* categories_ptr):
        # sequence_prefix is in reversed order
        cdef int i
        cdef double* probs = NULL
        cdef list arg = sequence_prefix
        try:
            # call function, convert result
            ret = self.pyfun_get_log_probs(arg[::-1])
            # If None is returned the ned of the sequence is reached
            if ret is None:
                categories_ptr[0] = 0
                return NULL
            # Perhaps this could be buffered somewhere?
            # Allocate C memory for it
            probs = <double*> PyMem_Malloc(sizeof(double) * len(ret))
            if probs == NULL:
                categories_ptr[0] = 0
                raise MemoryError()
            for i in range(len(ret)):
                probs[i] = ret[i]
            categories_ptr[0] = len(ret)
            return probs
        except:
            # catch any Python errors and return NULL
            categories_ptr[0] = 0
            return NULL
    cdef ur_node_t* get_state(self):
        return self.root
    cdef int get_max_categories(self):
        return self.max_categories
    cdef mt19937_64* get_generator(self):
        return &self.generator
    cpdef bool is_exhausted(self):
        return ur_is_exhausted(self.root)
    def __dealloc__(self):
        ur_free_all(self.root)
