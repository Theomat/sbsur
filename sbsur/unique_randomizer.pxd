# distutils: language=c++
from libcpp.vector cimport vector

# With a 'ctypedef' this struct should be referred as 'ur_node_t' instead of 'struct ur_node_t'
ctypedef struct ur_node_t

# Allocates an empty node
cdef ur_node_t *ur_new()
# Allocates en empty node with given probabilities
cdef ur_node_t *ur_new_with_log_probs(double* log_probs, size_t size)
# Free the specified node (should it free its children? no because the children should be already freed when called)
cdef ur_free(ur_node_t* node)

# True if we have created the specified child
cdef bool ur_is_child_expanded(ur_node_t *parent, int child_index)
# Create the specified child with the given log probs
cdef ur_node_t *ur_expand_node(ur_node_t *parent, double *log_probs, size_t categories, int child_index)
# Simple getter
cdef ur_node_t* ur_get_children(ur_node_t* node, int child_index)
cdef size_t ur_get_categories(ur_node_t *node)
# Return true => this node can be freed
cdef bool ur_is_exhausted(ur_node_t* node)
# Copy the current state of the log probs into out
cdef ur_get_log_probs(ur_node_t * node, double* out)

# Update probabilities after seeing sequence
cdef mark_sequence_sampled(ur_node_t* node, vector[int] sequence)


# Additional getters that might be useful
cdef int ur_get_number_of_sequences_sampled(ur_node_t *node)

#endif