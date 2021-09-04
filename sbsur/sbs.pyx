# distutils: language=c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.math cimport log, exp, fabs, fmax
from sequence_generator cimport SequenceGenerator
from unique_randomizer cimport ur_node_t, ur_is_exhausted, ur_get_log_probs, ur_get_parent, ur_is_leaf, ur_get_child, ur_get_index_in_parent, ur_get_categories, ur_mark_sampled, ur_has_parent, ur_get_possibles, ur_is_terminal, ur_is_child_expanded, ur_expand_node, ur_add_terminal_node
# Use the cython ones, they are thread-safe and give stats to python memory manager while behaving like C-ones (no GIL)
from cpython.mem cimport PyMem_Malloc, PyMem_Free
# Thread safe PRNG (they wrap to C++ standard lib)
from random_wrapper cimport uniform_real_distribution, mt19937_64

# Trick to allow compilation:
#   cython does not compile vector[ur_node_t*] but it compiles vector[ur_node_t_ptr]
ctypedef ur_node_t* ur_node_ptr

cdef vector[int] build_sequence(ur_node_t* leaf):
    cdef vector[int] sequence
    cdef ur_node_t* current = leaf
    while ur_has_parent(current):
        sequence.push_back(ur_get_index_in_parent(current))
        current = ur_get_parent(current)
    return sequence

cdef vector[int] build_sequence_with_end(ur_node_t* leaf, int end):
    cdef vector[int] sequence
    sequence.push_back(end)
    cdef ur_node_t* current = leaf
    while ur_has_parent(current):
        sequence.push_back(ur_get_index_in_parent(current))
        current = ur_get_parent(current)
    return sequence

cdef class GumbelHeap:
    cdef vector[ur_node_ptr] nodes
    cdef vector[double] log_probs
    cdef vector[double] gumbels
    cdef vector[int] child_indices
    cdef int index
    cdef readonly int size
    cdef readonly double min

    def __cinit__(self):
        self.nodes = vector[ur_node_ptr]()
        self.log_probs = vector[double]()
        self.gumbels = vector[double]()
        self.child_indices = vector[int]()
        self.index = 0
        self.size = 0
        self.min = -9999999999.0

    cdef void reserve(self, int size):
        self.nodes.reserve(size)
        self.log_probs.reserve(size)
        self.gumbels.reserve(size)
        self.child_indices.reserve(size)

    cdef void push_all(self, vector[ur_node_ptr]* nodes, vector[double]* log_probs, vector[double]* gumbels):
        cdef int i
        # This could be optimized by only percolating up the leaves
        for i in range(nodes[0].size()):
            self.push(nodes[0][i], log_probs[0][i], gumbels[0][i], -1)

    cdef void push_and_discard(self, ur_node_t* node, double logprob, double gumbel, int child_index):
        self.nodes[0] = node
        self.log_probs[0] = logprob
        self.gumbels[0] = gumbel
        self.child_indices[0] = child_index
        self.__percolate_down__(0)

    cdef void push(self, ur_node_t* node, double logprob, double gumbel, int child_index):
        self.nodes.push_back(node)
        self.log_probs.push_back(logprob)
        self.gumbels.push_back(gumbel)
        self.child_indices.push_back(child_index)

        self.size += 1
        self.__percolate_up__(self.nodes.size() - 1)

    cdef void __percolate_down__(self, int index):
        cdef int size = self.nodes.size()
        cdef int i = index
        cdef ur_node_t* node = self.nodes[i]
        cdef double logprob = self.log_probs[i]
        cdef double gumbel = self.gumbels[i]
        cdef int child_index = self.child_indices[i]

        cdef int left = 2 * i + 1
        cdef int right = 2 * i + 2
        while left < size:
            if right < size:
                if gumbel > self.gumbels[left]:
                    if gumbel > self.gumbels[right]:
                        # Both are possible
                        # Chose right arbitrarily
                        self.nodes[i] = self.nodes[right]
                        self.log_probs[i] = self.log_probs[right]
                        self.gumbels[i] = self.gumbels[right]
                        self.child_indices[i] = self.child_indices[right]
                        i = right
                    else:
                        self.nodes[i] = self.nodes[left]
                        self.log_probs[i] = self.log_probs[left]
                        self.gumbels[i] = self.gumbels[left]
                        self.child_indices[i] = self.child_indices[left]
                        i = left
                elif gumbel > self.gumbels[right]:
                    self.nodes[i] = self.nodes[right]
                    self.log_probs[i] = self.log_probs[right]
                    self.gumbels[i] = self.gumbels[right]
                    self.child_indices[i] = self.child_indices[right]
                    i = right
                else:
                    break
            else:
                if gumbel > self.gumbels[left]:
                    self.nodes[i] = self.nodes[left]
                    self.log_probs[i] = self.log_probs[left]
                    self.gumbels[i] = self.gumbels[left]
                    self.child_indices[i] = self.child_indices[left]
                    i = left
                else:
                    break
            # Update child index
            left = 2 * i + 1
            right = 2 * i + 2
            
        if i != index:
            self.nodes[i] = node
            self.log_probs[i] = logprob
            self.gumbels[i] = gumbel
            self.child_indices[i] = child_index
        self.min = self.gumbels[0]

    cdef void __percolate_up__(self, int index):
        cdef int i = index
        cdef ur_node_t* node = self.nodes[i]
        cdef double logprob = self.log_probs[i]
        cdef double gumbel = self.gumbels[i]
        cdef int child_index = self.child_indices[i]
        while i > 0 and gumbel < self.gumbels[(i - 1) // 2]:
            self.nodes[i] = self.nodes[(i - 1)  // 2]
            self.log_probs[i] = self.log_probs[(i - 1)  // 2]
            self.gumbels[i] = self.gumbels[(i - 1)  // 2]
            self.child_indices[i] = self.child_indices[(i - 1)  // 2]
            i = (i - 1) // 2 
        if i != index:
            self.nodes[i] = node
            self.log_probs[i] = logprob
            self.gumbels[i] = gumbel
            self.child_indices[i] = child_index
        self.min = self.gumbels[0]

    cdef ur_node_t* iterate(self, double* logprob_ptr, double* gumbel_ptr, int* child_index_ptr):
        # The [0] is the trick to replace the C * operator
        logprob_ptr[0] = self.log_probs[self.index]
        gumbel_ptr[0] = self.gumbels[self.index]
        child_index_ptr[0] = self.child_indices[self.index]
        self.index += 1
        return self.nodes[self.index - 1]

    cdef void reset(self):
        self.index = 0
        self.nodes.clear()
        self.log_probs.clear()
        self.gumbels.clear()
        self.child_indices.clear()
        self.size = 0
        self.min = -9999999999.0



cdef double sample_gumbels(double translate_logprob, double target_max, int nb_children, bool* possibles, double* logprobs, double* gumbels, uniform_real_distribution[double]* dist, mt19937_64 *gen):
    cdef double max_gumbel = -9999999999.0
    cdef int i
    for i in range(nb_children):
        if possibles[i] == 0:
            continue
        gumbels[i] = logprobs[i] + translate_logprob - log(-log(dist[0](gen[0])))
        if gumbels[i] > max_gumbel:
            max_gumbel = gumbels[i]
    # Use equations (23) and (24) in Appendix B.3 of the SBS paper.
    cdef double v = 0
    for i in range(nb_children):
        if possibles[i] == 0:
            continue
        if gumbels[i] >= max_gumbel:
            gumbels[i] = target_max
        else:
            v = target_max - gumbels[i] + log(1.0 - exp(gumbels[i] - max_gumbel))
            gumbels[i] = target_max - fmax(v, 0.0) - log(1.0 + exp(-fabs(v)))

cdef vector[(vector[int], double)] c_sample(SequenceGenerator generator, int batch_size):
    cdef vector[(vector[int], double)] out
    cdef ur_node_t* root = generator.get_state()
    if ur_is_exhausted(root) or batch_size <= 0:
        return out
    cdef uniform_real_distribution[double] dist = uniform_real_distribution[double](0.0,1.0)
    cdef mt19937_64* gen = generator.get_generator()

    # Current state
    cdef vector[ur_node_ptr] internal
    cdef vector[double] internal_log_probs
    cdef vector[double] internal_gumbels
    internal.reserve(batch_size)
    internal_log_probs.reserve(batch_size)
    internal_gumbels.reserve(batch_size)

    # Next expansion state
    cdef GumbelHeap heap = GumbelHeap.__new__(GumbelHeap)
    heap.reserve(batch_size)
    # Leaves
    cdef vector[ur_node_ptr] leaves
    cdef vector[double] leaves_logprobs
    cdef vector[double] leaves_gumbels
    leaves.reserve(batch_size)
    leaves_logprobs.reserve(batch_size)
    leaves_gumbels.reserve(batch_size)

    # Nodes to expand with get_logprobs
    cdef vector[ur_node_ptr] to_expand_nodes
    cdef vector[float] to_expand_gumbels
    cdef vector[float] to_expand_logprob
    cdef vector[int] to_expand_child_indices
    cdef vector[vector[int]] sequences
    to_expand_nodes.reserve(batch_size)
    to_expand_gumbels.reserve(batch_size)
    to_expand_logprob.reserve(batch_size)
    to_expand_child_indices.reserve(batch_size)
    sequences.reserve(batch_size)

    # Initialisation
    internal.push_back(root)
    internal_log_probs.push_back(0.0)
    internal_gumbels.push_back(0.0)

    # Loop variables
    cdef ur_node_t* current
    cdef ur_node_t* child_node
    cdef double current_log_prob
    cdef double current_gumbel 
    cdef int child_index
    cdef int nb_children
    cdef int i
    # Buffers
    cdef double* logprobs
    cdef bool* possibles
    cdef double* buffer_gumbels = <double*> PyMem_Malloc(sizeof(double) * generator.get_max_categories())

    cdef double** new_node_logprobs
    cdef double* new_node_logprob
    cdef int* new_node_categories = <int*> PyMem_Malloc(sizeof(int) * batch_size)
    cdef int new_node_category

    while not internal.empty():
        # Add leaves to next_nodes
        heap.push_all(&leaves, &leaves_logprobs, &leaves_gumbels)
        # Clear leaves
        leaves.clear()
        leaves_logprobs.clear()
        leaves_gumbels.clear()

        # Expand one depth level deeper
        while not internal.empty():
            # Take one internal node
            current = internal.back()
            internal.pop_back()
            current_log_prob = internal_log_probs.back()
            internal_log_probs.pop_back()
            current_gumbel = internal_gumbels.back()
            internal_gumbels.pop_back()
            # Get number of children
            nb_children = ur_get_categories(current)
            # Get the log probs
            logprobs = ur_get_log_probs(current)
            # Get possibles
            possibles = ur_get_possibles(current)

            # Fill buffer_gumbels with the appropriate data
            sample_gumbels(current_log_prob, current_gumbel, nb_children, possibles, logprobs, buffer_gumbels, &dist, gen)
            # Update candidates
            for i in range(nb_children):
                if possibles[i] == 0 or (heap.size >= batch_size and heap.min >= buffer_gumbels[i]):
                    continue 
                
                # Change data if child exist or not
                child_index = i
                child_node = current
                if ur_is_child_expanded(current, i):
                    child_index = -1
                    child_node = ur_get_child(current, i)
                # Add candidate
                # Not enough internal candidates yet
                if heap.size < batch_size:
                    heap.push(child_node, logprobs[i], buffer_gumbels[i], child_index)
                else:
                    # So we whould discard the min
                    heap.push_and_discard(child_node, logprobs[i], buffer_gumbels[i], child_index)       
        
        # Move from heap to leaves and internal
        for _ in range(heap.size):
            # Pop from heap
            current = heap.iterate(&current_log_prob, &current_gumbel, &child_index)

            # Expand child if it does not exist
            if child_index > -1:
                to_expand_nodes.push_back(current)
                sequences.push_back(build_sequence_with_end(current, child_index))
                to_expand_gumbels.push_back(current_gumbel)
                to_expand_logprob.push_back(current_log_prob)
                to_expand_child_indices.push_back(child_index)
            else:
                # Sort in group for next iteration 
                if ur_is_terminal(current):
                    # Add to leaves
                    leaves.push_back(current)
                    leaves_logprobs.push_back(current_log_prob)
                    leaves_gumbels.push_back(current_gumbel)
                else:
                    internal.push_back(current)
                    internal_log_probs.push_back(current_log_prob)
                    internal_gumbels.push_back(current_gumbel)

        # Segfault occurs here
        if sequences.size() > 0:
            # Get new logprobs
            new_node_logprobs = generator.get_log_probs(&sequences, new_node_categories)
            # Expand one depth level deeper all nodes that need to be
            for i in range(to_expand_nodes.size()):
   
                # Get current values
                current = to_expand_nodes[i]
                current_log_prob = to_expand_logprob[i]
                current_gumbel = to_expand_gumbels[i]
                child_index = to_expand_child_indices[i]
                new_node_logprob = new_node_logprobs[i]
                new_node_category = new_node_categories[i]

                if new_node_category == 0:
                    # There is no sequence afterwards
                    ur_add_terminal_node(current, child_index) 
                    current = ur_get_child(current, child_index)
                    # Add to leaves
                    leaves.push_back(current)
                    leaves_logprobs.push_back(current_log_prob)
                    leaves_gumbels.push_back(current_gumbel)                   
                else:
                    ur_expand_node(current, new_node_logprob, new_node_category, child_index)

                    current = ur_get_child(current, child_index)
                    internal.push_back(current)
                    internal_log_probs.push_back(current_log_prob)
                    internal_gumbels.push_back(current_gumbel)

            to_expand_nodes.clear()
            to_expand_gumbels.clear()
            to_expand_child_indices.clear()
            to_expand_logprob.clear()
            sequences.clear()

        # Reset the heap
        heap.reset()
   
    # Free allocated memory
    PyMem_Free(buffer_gumbels)
    PyMem_Free(new_node_categories)

    # Build sampled sequences
    cdef ur_node_t* leaf
    for i in range(leaves.size()):
        leaf = leaves[i]
        # Build sequence
        out.push_back((build_sequence(leaf), leaves_gumbels[i]))
        # Mark sampled
        ur_mark_sampled(leaf)

    return out


def sample(generator: SequenceGenerator, batch_size: int) -> list:
    cdef vector[(vector[int], double)] sampled = c_sample(generator, batch_size)
    cdef list sequences = []
    cdef (vector[int], double) element
    cdef vector[int] seq
    cdef double gumbel
    for element in sampled:
        seq, gumbel = element
        sequences.append((seq, gumbel))
    # Sort them according to sampling order
    sequences.sort(key=lambda s:s[1], reverse=True)
    cdef list output = [x[0][::-1] for x in sequences]
    return output