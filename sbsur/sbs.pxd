# distutils: language=c++
from libcpp.vector cimport vector

from sequence_generator cimport SequenceGenerator

cdef vector<vector<int>> c_sample(SequenceGenerator generator, int batch_size)