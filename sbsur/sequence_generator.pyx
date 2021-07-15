# distutils: language=c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
from libcpp.vector cimport vector
# Use the cython ones, they are thread-safe and give stats to python memory manager while behaving like C-ones (no GIL)
from cpython.mem cimport PyMem_Malloc
from random_wrapper cimport mt19937

from unique_randomizer cimport ur_node_t, ur_free_all, ur_new


cdef class SequenceGenerator:

    def __cinit__(self, python_callback, int max_categories, seed):
        self.pyfun_get_log_probs = <void*>python_callback
        self.root = ur_new()
        self.max_categories = max_categories
        if seed is None:
            self.generator = mt19937()
        else:
            self.generator = mt19937(seed)

    cdef double* get_log_probs(self, vector[int] sequence_prefix, int* categories_ptr):
        try:
            # recover Python function object from void* argument
            func = <object>self.pyfun_get_log_probs
            # call function, convert result into 0/1 for True/False
            ret = func([x for x in sequence_prefix])
            # If None is returned the ned of the sequence is reached
            if ret is None:
                categories_ptr[0] = 0
                return NULL
            # Perhaps this could be buffered somewhere?
            # Allocate C memory for it
            probs = <double*> PyMem_Malloc(sizeof(double) * len(ret))
            if not probs:
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
    cdef mt19937 get_generator(self):
        return self.generator
    def __dealloc__(self):
        ur_free_all(self.root)
