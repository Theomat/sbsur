# distutils: language=c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.math cimport log, exp
from sequence_generator cimport SequenceGenerator
from unique_randomizer cimport ur_node_t, ur_is_exhausted, ur_get_log_probs, ur_get_parent, ur_is_leaf, ur_get_child, ur_get_index_in_parent, ur_get_categories, ur_mark_sampled, ur_has_parent, ur_get_possibles, ur_is_terminal, ur_is_child_expanded, ur_expand_node, ur_add_terminal_node
# Use the cython ones, they are thread-safe and give stats to python memory manager while behaving like C-ones (no GIL)
from cpython.mem cimport PyMem_Malloc, PyMem_Free
# Thread safe PRNG (they wrap to C++ standard lib)
from random_wrapper cimport uniform_real_distribution, mt19937

# Trick to allow compilation:
#   cython does not compile vector[ur_node_t*] but it compiles vector[ur_node_t_ptr]
ctypedef ur_node_t* ur_node_ptr


cdef vector[int] build_sequence(ur_node_t* leaf):
    cdef vector[int] sequence = vector[int]()
    cdef ur_node_t* current = leaf
    while ur_has_parent(current):
        sequence.push_back(ur_get_index_in_parent(current))
        current = ur_get_parent(current)
    return sequence

cdef class GumbelHeap:
    cdef vector[ur_node_ptr] nodes
    cdef vector[double] log_probs
    cdef vector[double] gumbels
    cdef int index

    def __cinit__(self):
        self.nodes = vector[ur_node_ptr]()
        self.log_probs = vector[double]()
        self.gumbels = vector[double]()
        self.index = 0

    cdef int size(self):
        return self.nodes.size()

    cdef void push_all(self, vector[ur_node_ptr] nodes, vector[double] log_probs, vector[double] gumbels):
        cdef int i
        # This could be optimized by only percolating up the leaves
        for i in range(nodes.size()):
            self.push(nodes[i], log_probs[i], gumbels[i])

    cdef void push_and_discard(self, ur_node_t* node, double logprob, double gumbel):
        self.nodes[0] = node
        self.log_probs[0] = logprob
        self.gumbels[0] = gumbel
        self.__percolate_down__(0)

    cdef void push(self, ur_node_t* node, double logprob, double gumbel):
        self.nodes.push_back(node)
        self.log_probs.push_back(logprob)
        self.gumbels.push_back(gumbel)
        self.__percolate_up__(self.nodes.size() - 1)

    cdef void __percolate_down__(self, int index):
        cdef int size = self.nodes.size()
        cdef int i = index
        cdef ur_node_t* node = self.nodes[i]
        cdef double logprob = self.log_probs[i]
        cdef double gumbel = self.gumbels[i]

        cdef int left = 2 * i + 1
        cdef int right = 2 * i + 2
        while i < size // 2:
            left = 2 * i + 1
            right = 2 * i + 2
            if left < size and right < size:
                if self.gumbels[left] < gumbel:
                    # Chose right arbitrarily
                    if self.gumbels[right] < gumbel:
                        self.nodes[i] = self.nodes[right]
                        self.log_probs[i] = self.log_probs[right]
                        self.gumbels[i] = self.gumbels[right]
                        i = right
                    else:
                        self.nodes[i] = self.nodes[left]
                        self.log_probs[i] = self.log_probs[left]
                        self.gumbels[i] = self.gumbels[left]
                        i = left
                elif self.gumbels[right] < gumbel:
                    self.nodes[i] = self.nodes[right]
                    self.log_probs[i] = self.log_probs[right]
                    self.gumbels[i] = self.gumbels[right]
                    i = right
                else:
                    break
            elif left < size and self.gumbels[left] < gumbel:
                self.nodes[i] = self.nodes[left]
                self.log_probs[i] = self.log_probs[left]
                self.gumbels[i] = self.gumbels[left]
                i = left
            else:
                break
            
        if i != index:
            self.nodes[i] = node
            self.log_probs[i] = logprob
            self.gumbels[i] = gumbel

    cdef void __percolate_up__(self, int index):
        cdef int i = index
        cdef ur_node_t* node = self.nodes[i]
        cdef double logprob = self.log_probs[i]
        cdef double gumbel = self.gumbels[i]
        while i > 0 and gumbel < self.gumbels[i // 2]:
            self.nodes[i] = self.nodes[i // 2]
            self.log_probs[i] = self.log_probs[i // 2]
            self.gumbels[i] = self.gumbels[i // 2]
            i = i // 2 
        if i != index:
            self.nodes[i] = node
            self.log_probs[i] = logprob
            self.gumbels[i] = gumbel

    cdef ur_node_t* iterate(self, double* logprob_ptr, double* gumbel_ptr):
        cdef ur_node_t* node = self.nodes[self.index]
        # The [0] is the tirck to replace the C * operator
        logprob_ptr[0] = self.log_probs[self.index]
        gumbel_ptr[0] = self.gumbels[self.index]
        self.index += 1
        return node

    cdef void reset(self):
        self.index = 0
        self.nodes.clear()
        self.log_probs.clear()
        self.gumbels.clear()

    cdef double min(self):
        if self.nodes.size() == 0:
            return -9999999999.0
        return self.gumbels[0]

cdef void update_with_candidate(GumbelHeap heap, ur_node_t* candidate, double logprob, double gumbel, int batch_size):
    # Not enough internal candidates yet
    if heap.size() < batch_size:
        heap.push(candidate, logprob, gumbel)
        return
    # So we whould discard the min
    heap.push_and_discard(candidate, logprob, gumbel)

cdef bool should_add_candidate(GumbelHeap heap, double gumbel, int batch_size):
    return heap.size() < batch_size or heap.min() < gumbel

cdef double sample_gumbels(double target_max, int nb_children, bool* possibles, double* logprobs, double* gumbels, uniform_real_distribution[double] dist, mt19937 gen):
    cdef double max_gumbel = -9999999999.0
    cdef int i
    for i in range(nb_children):
        if not possibles[i]:
            continue
        gumbels[i] = logprobs[i] - log(-log(dist(gen)))
        if gumbels[i] > max_gumbel:
            max_gumbel = gumbels[i]
    # Use equations (23) and (24) in Appendix B.3 of the SBS paper.
    cdef double v = 0
    for i in range(nb_children):
        if not possibles[i]:
            continue
        v = target_max - gumbels[i]
        if gumbels[i] >= max_gumbel:
            v = 0
        else:
            v += log(1 - exp(gumbels[i] - max_gumbel))
        gumbels[i] = target_max - max(v, 0)
        if gumbels[i] <= max_gumbel:
            gumbels[i] -= log(1 + exp(-abs(v)))

cdef vector[(vector[int], double)] c_sample(SequenceGenerator generator, int batch_size):
    cdef vector[(vector[int], double)] out = [] #vector[(vector[int], float)](batch_size) doesn't work and can't be instancied
    cdef ur_node_t* root = generator.get_state()
    if ur_is_exhausted(root) or batch_size <= 0:
        return out
    cdef uniform_real_distribution[double] dist = uniform_real_distribution[double](0.0,1.0)
    cdef mt19937 gen = generator.get_generator()

    # Current state
    cdef vector[ur_node_ptr] internal = vector[ur_node_ptr]()
    cdef vector[double] internal_log_probs = vector[double]()
    cdef vector[double] internal_gumbels = vector[double]()
    # Next expansion state
    cdef GumbelHeap heap = GumbelHeap.__new__(GumbelHeap)
    # Leaves
    cdef vector[ur_node_ptr] leaves = vector[ur_node_ptr]()
    cdef vector[double] leaves_logprobs = vector[double]()
    cdef vector[double] leaves_gumbels = vector[double]()

    # Initialisation
    internal.push_back(root)
    internal_log_probs.push_back(0.0)
    internal_gumbels.push_back(0.0)

    # Loop variables
    cdef ur_node_t* current
    cdef double current_log_prob
    cdef double current_gumbel 
    cdef int nb_children
    cdef double logprob
    cdef double gumbel
    cdef int i
    cdef vector[int] sequence
    # Buffers
    cdef double* buffer_logprobs = <double*> PyMem_Malloc(sizeof(double) * generator.get_max_categories())
    cdef double* buffer_gumbels = <double*> PyMem_Malloc(sizeof(double) * generator.get_max_categories())
    cdef bool* possibles = <bool*> PyMem_Malloc(sizeof(bool) * generator.get_max_categories())

    cdef double* new_node_logprobs
    cdef int new_node_categories

    while not internal.empty():
        # Add leaves to next_nodes
        heap.push_all(leaves, leaves_logprobs, leaves_gumbels)

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
            # Get the log probs in buffer_logprobs
            ur_get_log_probs(current, buffer_logprobs)
            # Get possibles
            ur_get_possibles(current, possibles)
            # Add current node log prob to children
            for i in range(nb_children):
                buffer_logprobs[i] += current_log_prob

            # Fill buffer_gumbels with the appropriate data
            sample_gumbels(current_gumbel, nb_children, possibles, buffer_logprobs, buffer_gumbels, dist, gen)

            # Update candidates
            for i in range(nb_children):
                if not possibles[i] or not should_add_candidate(heap, buffer_gumbels[i], batch_size):
                    continue 
                
                # Check if child exists and creates it if it doesn't
                if not ur_is_child_expanded(current, i):
                    # Get new logprobs
                    sequence = build_sequence(current)
                    sequence.push_back(i)
                    new_node_logprobs = generator.get_log_probs(sequence, &new_node_categories)
                    if new_node_categories == 0:
                        # There is no sequence afterwards
                        ur_add_terminal_node(current, i)                    
                    else:
                        ur_expand_node(current, new_node_logprobs, new_node_categories, i)

                update_with_candidate(heap, ur_get_child(current, i), buffer_logprobs[i], buffer_gumbels[i], batch_size)

        # Move from heap to leaves and internal
        for _ in range(heap.size()):
            # Pop from heap
            current = heap.iterate(&logprob, &gumbel)
            if ur_is_terminal(current):
                # Add to leaves
                leaves.push_back(current)
                leaves_logprobs.push_back(logprob)
                leaves_gumbels.push_back(gumbel)
            else:
                internal.push_back(current)
                internal_log_probs.push_back(logprob)
                internal_gumbels.push_back(gumbel)
        # Reset the heap
        heap.reset()
   
    # Free allocated memory
    PyMem_Free(buffer_logprobs)
    PyMem_Free(buffer_gumbels)
    PyMem_Free(possibles)

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
    sequences = []
    cdef (vector[int], double) element
    cdef vector[int] rseq
    cdef double gumbel
    cdef list seq
    cdef int el
    for element in sampled:
        rseq, gumbel = element
        seq = []
        for el in rseq:
            seq.append(el)
        sequences.append((seq, gumbel))
    # Sort them according to sampling order
    sequences.sort(key=lambda s:s[1], reverse=True)
    output = [x[0] for x in sequences]
    return output