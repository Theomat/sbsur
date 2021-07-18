# distutils: language=c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
from libcpp cimport bool
from libc.math cimport log, exp

from libc.string cimport memcpy 
# Use the cython ones, they are thread-safe and give stats to python memory manager while behaving like C-ones (no GIL)
from cpython.mem cimport PyMem_Malloc, PyMem_Free

# Allocates an empty node
cdef ur_node_t *ur_new():
    cdef ur_node_t* node = <ur_node_t*>PyMem_Malloc(sizeof(ur_node_t))
    node.categories = -1
    node.terminal = 0
    node.logprobs = NULL
    node.parent = NULL
    node.children = NULL
    node.possibles = NULL
    node.original_logprobs = NULL
    node.normed = 0
    node.eventualities = 0
    node.unsampled_fraction = 1.0
    return node
# Free the specified node (should it free its children? no because the children should be already freed when called)
cdef void ur_free(ur_node_t* node):
    # Assume: all my children should already be freed
    if node.logprobs != NULL:
        PyMem_Free(node.logprobs)
    if node.original_logprobs != NULL:
        PyMem_Free(node.original_logprobs)
    if node.possibles != NULL:
        PyMem_Free(node.possibles)
    if node.children != NULL:
        PyMem_Free(node.children)
    PyMem_Free(node)
# Free the specified node and all of its children
cdef void ur_free_all(ur_node_t* node):
    cdef int i
    # Free all children
    if node.children != NULL:
        for i in range(node.categories):
            if node.children[i] != NULL:
                ur_free_all(node.children[i])
    # Now classic free
    ur_free(node)

cdef void set_child(ur_node_t* parent, ur_node_t* child, int child_index):
    child.parent = parent
    child.index_in_parent = child_index
    # Check children has been allocated
    cdef int i
    if parent.children == NULL:
        parent.children = <ur_node_t**>PyMem_Malloc(sizeof(ur_node_t*) * parent.categories)
        for i in range(parent.categories):
            parent.children[i] = NULL
    # Set child
    parent.children[child_index] = child
# True if we have created the specified child
cdef bool ur_is_child_expanded(ur_node_t *parent, int child_index):
    return parent.children != NULL and parent.children[child_index] != NULL
# Should only be used for the root
cdef void ur_set_logprobs(ur_node_t *node, double *log_probs, int categories):
    node.logprobs = <double*>PyMem_Malloc(sizeof(double) * categories)
    memcpy(node.logprobs, log_probs, sizeof(double) * categories)
    node.original_logprobs = log_probs
    node.categories = categories
    node.normed = 1
    node.eventualities = categories
    cdef int i
    node.possibles = <bool*>PyMem_Malloc(sizeof(bool) * categories)
    for i in range(categories):
        node.possibles[i] = 1
# Create the specified child with the given log probs
cdef void ur_expand_node(ur_node_t *parent, double *log_probs, int categories, int child_index):
    cdef ur_node_t* child = ur_new()
    ur_set_logprobs(child, log_probs, categories)
    # Set child
    set_child(parent, child, child_index)   
   
# Add a new terminal child
cdef void ur_add_terminal_node(ur_node_t *parent, int child_index):
    cdef ur_node_t* child = ur_new()
    set_child(parent, child, child_index)
    ur_mark_terminal(child)

# Simple getter
cdef ur_node_t *ur_get_child(ur_node_t* node, int child_index):
    return node.children[child_index]
cdef int ur_get_index_in_parent(ur_node_t* node):
    return node.index_in_parent
cdef ur_node_t *ur_get_parent(ur_node_t* node):
    return node.parent
cdef int ur_get_categories(ur_node_t *node):
    return node.categories
# Return 1 => this node can be freed
cdef bool ur_is_exhausted(ur_node_t* node):
    return node.unsampled_fraction <= 0
# Return an array in which: array[i] == 1 iff the category 'i' can be sampled (this has no link to if there is an actual childe node, we are only interested in the fact that it can be sampled) 
cdef void ur_get_possibles(ur_node_t* node, bool* out):
    memcpy(out, node.possibles, sizeof(bool) * node.categories)

# A leaf is a node that has yet to be sampled or a terminal e.g. when created a node is a leaf because it doesn't have any children, but it's not a terminal because it hasn't been marked has such
cdef bool ur_is_leaf(ur_node_t* node):
    return node.terminal == 1 or node.children == NULL
# A terminal is a node that has no child and will have no child, a terminal node can't be sampled
cdef bool ur_is_terminal(ur_node_t* node):
    return node.terminal == 1
cdef void ur_mark_terminal(ur_node_t* node):
    node.terminal = 1
    exhaust(node)
    node.unsampled_fraction = 1.0
cdef bool ur_has_parent(ur_node_t* node):
    return node.parent != NULL
# Copy the current state of the log probs into out
cdef void ur_get_log_probs(ur_node_t * node, double* out):
    cdef double psum = 0
    cdef int i = 0
    if node.normed == 0:
        # Normalize
        psum = log(node.unsampled_fraction)
        for i in range(node.categories):
            if node.possibles[i] == 1:
                node.logprobs[i] = node.original_logprobs[i] - psum
        node.normed = 1
    memcpy(out, node.logprobs, sizeof(double) * node.categories)

cdef void exhaust(ur_node_t* node):
    node.unsampled_fraction = 0
    node.categories = 0
    node.eventualities = 0
    if node.logprobs != NULL:
        PyMem_Free(node.logprobs)
        node.logprobs = NULL
    if node.original_logprobs != NULL:
        PyMem_Free(node.original_logprobs)
        node.original_logprobs = NULL
    if node.possibles != NULL:
        PyMem_Free(node.possibles)
        node.possibles = NULL
    if node.children != NULL:
        PyMem_Free(node.children)
        node.children = NULL
# Update probabilities after seeing sequence stopped at the resulting leaf node
cdef void ur_mark_sampled(ur_node_t* leaf):
    # I am a terminal node (a placeholder like)
    # So of course I have a parent
    leaf.unsampled_fraction = 0
    mark_children_sampled(leaf.parent, leaf.index_in_parent)

cdef void mark_children_sampled(ur_node_t* parent, int child_index):
    cdef double child_unsampled = parent.children[child_index].unsampled_fraction
    cdef int i = 0
    parent.normed = 0
    if child_unsampled <= 0:
        # No longer possible
        parent.possibles[child_index] = 0
        parent.eventualities -= 1

        # Free child node
        ur_free(parent.children[child_index])
        parent.children[child_index] = NULL

        # if no more eventualities, exhaust the node
        if parent.eventualities <= 0:
            exhaust(parent)
        else:
            parent.unsampled_fraction = 0
            for i in range(parent.categories):
                if parent.possibles[i] == 1:
                    parent.unsampled_fraction += exp(parent.logprobs[i])
    else:
        parent.unsampled_fraction -= exp(parent.logprobs[child_index])
        parent.logprobs[child_index] = log(child_unsampled) + parent.original_logprobs[child_index]
        parent.unsampled_fraction += exp(parent.logprobs[child_index])

    # Back propagate
    if parent.parent != NULL:
        mark_children_sampled(parent.parent, parent.index_in_parent)
    

