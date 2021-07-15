# distutils: language=c++
from libcpp.vector cimport vector
from libcpp cimport bool

# Why do we need a struct? So that we can manage the memory as we want
# should be referred as 'ur_node_t' instead of 'struct ur_node_t'
cdef struct ur_node_t

# Allocates an empty node
cdef ur_node_t *ur_new() nogil
# Free the specified node (should it free its children? no because the children should be already freed when called)
cdef ur_free(ur_node_t* node) nogil
# Free the specified node and all of its children
cdef ur_free_all(ur_node_t* node) nogil

# True if we have created the specified child
cdef bool ur_is_child_expanded(ur_node_t *parent, int child_index) nogil
# Create the specified child with the given log probs
cdef ur_expand_node(ur_node_t *parent, double *log_probs, int categories, int child_index) nogil
# Add a new terminal child
cdef ur_add_terminal_node(ur_node_t *parent, int child_index) nogil
# Simple getter
cdef ur_node_t *ur_get_child(ur_node_t* node, int child_index) nogil
cdef int ur_get_index_in_parent(ur_node_t* node) nogil
cdef ur_node_t *ur_get_parent(ur_node_t* node) nogil
cdef int ur_get_categories(ur_node_t *node) nogil
# Return true => this node can be freed
cdef bool ur_is_exhausted(ur_node_t* node) nogil
# Return an array in which: array[i] == true iff the category 'i' can be sampled (this has no link to if there is an actual childe node, we are only interested in the fact that it can be sampled) 
cdef ur_get_possibles(ur_node_t* node, bool* out) nogil
# A leaf is a node that has yet to be sampled or a terminal e.g. when created a node is a leaf because it doesn't have any children, but it's not a terminal because it hasn't been marked has such
cdef bool ur_is_leaf(ur_node_t* node) nogil
# A terminal is a node that has no child and will have no child, a terminal node can't be sampled
cdef bool ur_is_terminal(ur_node_t* node) nogil
cdef ur_mark_terminal(ur_node_t* node) nogil
cdef bool ur_has_parent(ur_node_t* node) nogil
# Copy the current state of the log probs into out
cdef ur_get_log_probs(ur_node_t * node, double* out) nogil

# Update probabilities after seeing sequence stopped at the resulting leaf node
cdef ur_mark_sampled(ur_node_t* leaf) nogil

#endif