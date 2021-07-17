# distutils: language=c++
from libcpp cimport bool

# Why do we need a struct? So that we can manage the memory as we want
# should be referred as 'ur_node_t' instead of 'struct ur_node_t'
cdef struct ur_node_t:
    double* original_logprobs
    double* logprobs
    bool* possibles
    int eventualities
    double unsampled_fraction
    bool normed
    int categories
    ur_node_t** children
    bool terminal
    ur_node_t* parent
    int index_in_parent

# Allocates an empty node
cdef ur_node_t *ur_new()
# Free the specified node (should it free its children? no because the children should be already freed when called)
cdef void ur_free(ur_node_t* node)
# Free the specified node and all of its children
cdef void ur_free_all(ur_node_t* node)

# True if we have created the specified child
cdef bool ur_is_child_expanded(ur_node_t *parent, int child_index)
# Should only be used for the root
cdef void ur_set_logprobs(ur_node_t *node, double *log_probs, int categories)
# Create the specified child with the given log probs
cdef void ur_expand_node(ur_node_t *parent, double *log_probs, int categories, int child_index)
# Add a new terminal child
cdef void ur_add_terminal_node(ur_node_t *parent, int child_index)
# Simple getter
cdef ur_node_t *ur_get_child(ur_node_t* node, int child_index)
cdef int ur_get_index_in_parent(ur_node_t* node)
cdef ur_node_t *ur_get_parent(ur_node_t* node)
cdef int ur_get_categories(ur_node_t *node)
# Return true => this node can be freed
cdef bool ur_is_exhausted(ur_node_t* node)
# Return an array in which: array[i] == true iff the category 'i' can be sampled (this has no link to if there is an actual childe node, we are only interested in the fact that it can be sampled) 
cdef void ur_get_possibles(ur_node_t* node, bool* out)
# A leaf is a node that has yet to be sampled or a terminal e.g. when created a node is a leaf because it doesn't have any children, but it's not a terminal because it hasn't been marked has such
cdef bool ur_is_leaf(ur_node_t* node)
# A terminal is a node that has no child and will have no child, a terminal node can't be sampled
cdef bool ur_is_terminal(ur_node_t* node)
cdef bool ur_has_parent(ur_node_t* node)
# Copy the current state of the log probs into out
cdef void ur_get_log_probs(ur_node_t * node, double* out)

# Update probabilities after seeing sequence stopped at the resulting leaf node
cdef void ur_mark_sampled(ur_node_t* leaf)
