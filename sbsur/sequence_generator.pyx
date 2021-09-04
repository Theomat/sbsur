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
        cdef vector[vector[int]] empty = [[]]
        cdef double** logprobs = self.get_log_probs(&empty, &categories)
        ur_set_logprobs(self.root, logprobs[0], categories)
        self.max_categories = max_categories
        cdef random_device r
        if seed is None:
            self.generator = mt19937_64(r())
        else:
            self.generator = mt19937_64(seed)

    cdef double** get_log_probs(self, vector[vector[int]]* sequence_prefixes, int* categories_ptr):
        # sequence_prefix is in reversed order
        cdef int prob_index
        cdef int index
        cdef double* probs = NULL
        cdef list arg = sequence_prefixes[0]
        cdef list logprobs = self.pyfun_get_log_probs([x[::-1] for x in arg])
        cdef double** output = <double**> PyMem_Malloc(sizeof(double*) * sequence_prefixes.size())
        for index in range(sequence_prefixes.size()):
            output[index] = NULL
            categories_ptr[index] = 0
            ret = logprobs[index]
            # If None is returned the end of the sequence is reached
            if ret is None or len(ret) == 0:
                continue
            # Allocate C memory for it
            probs = <double*> PyMem_Malloc(sizeof(double) * len(ret))
            if probs == NULL:
                raise MemoryError()
            for i in range(len(ret)):
                probs[i] = ret[i]

            output[index] = probs
            categories_ptr[index] = len(ret)

        return output
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
