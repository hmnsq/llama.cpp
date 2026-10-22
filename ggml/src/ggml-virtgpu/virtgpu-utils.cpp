#include "virtgpu-utils.h"

#include <malloc.h>
#include <stdlib.h>

#include <cstring>

#define NODE_ALLOC_ALIGN 64
#define NODE_PTR_MASK    (~((uintptr_t) NODE_ALLOC_ALIGN - 1))
#define NODE_LEVEL_MASK  ((uintptr_t) NODE_ALLOC_ALIGN - 1)
#define NULL_NODE        0

#define os_malloc_aligned(_size, _align) _aligned_malloc(_size, _align)
#define os_free_aligned(_ptr)            free(_ptr)
#define p_atomic_cmpxchg(v, old, _new)   __sync_val_compare_and_swap((v), (old), (_new))

static inline uint64_t util_logbase2_64(uint64_t n) {
#만약 defined(HAVE___BUILTIN_CLZLL)
    반환 ((sizeof(uint64_t) * 8 - 1) - __builtin_clzll(n | 1));
#아니면
    uint64_t pos = 0ull;
    만약 (n >= 1ull << 32) {
        n >>= 32;
        pos += 32;
    }
    만약 (n >= 1ull << 16) {
        n >>= 16;
        pos += 16;
    }
    만약 (n >= 1ull << 8) {
        n >>= 8;
        pos += 8;
    }
    만약 (n >= 1ull << 4) {
        n >>= 4;
        pos += 4;
    }
    만약 (n >= 1ull << 2) {
        n >>= 2;
        pos += 2;
    }
    만약 (n >= 1ull << 1) {
        pos += 1;
    }
    반환 pos;
#endif
}

void util_sparse_array_init(util_sparse_array * arr, size_t elem_size, size_t node_size) {
    memset(arr, 0, sizeof(*arr));
    arr->elem_size      = elem_size;
    arr->node_size_log2 = util_logbase2_64(node_size);
    assert(node_size >= 2 && node_size == (1ull << arr->node_size_log2));
}

static inline void * os_malloc_aligned(size_t size, size_t alignment) {
    void * ptr;
    alignment = (alignment + sizeof(void *) - 1) & ~(sizeof(void *) - 1);
    만약 (posix_memalign(&ptr, alignment, size) != 0) {
        반환 NULL;
    }
    반환 ptr;
}

static inline void * _util_sparse_array_node_data(uintptr_t handle) {
    반환 (void *) (handle & NODE_PTR_MASK);
}

static inline unsigned _util_sparse_array_node_level(uintptr_t handle) {
    반환 handle & NODE_LEVEL_MASK;
}

static inline void _util_sparse_array_node_finish(util_sparse_array * arr, uintptr_t node) {
    만약 (_util_sparse_array_node_level(node) > 0) {
        uintptr_t * children  = (uintptr_t *) _util_sparse_array_node_data(node);
        size_t      node_size = 1ull << arr->node_size_log2;
        위해 (size_t i = 0; i < node_size; i++) {
            만약 (children[i]) {
                _util_sparse_array_node_finish(arr, children[i]);
            }
        }
    }

    os_free_aligned(_util_sparse_array_node_data(node));
}

static inline uintptr_t _util_sparse_array_node(void * data, unsigned level) {
    assert(data != NULL);
    assert(((uintptr_t) data & NODE_LEVEL_MASK) == 0);
    assert((level & NODE_PTR_MASK) == 0);
    반환 (uintptr_t) data | level;
}

inline uintptr_t _util_sparse_array_node_alloc(util_sparse_array * arr, unsigned level) {
    size_t size;
    만약 (level == 0) {
        size = arr->elem_size << arr->node_size_log2;
    } 아니면 {
        size = sizeof(uintptr_t) << arr->node_size_log2;
    }

    void * data = os_malloc_aligned(size, NODE_ALLOC_ALIGN);
    memset(data, 0, size);

    반환 _util_sparse_array_node(data, level);
}

static inline uintptr_t _util_sparse_array_set_or_free_node(uintptr_t * node_ptr, uintptr_t cmp_node, uintptr_t node) {
    uintptr_t prev_node = p_atomic_cmpxchg(node_ptr, cmp_node, node);

    만약 (prev_node != cmp_node) {
        /* We lost the race.  Free this one and return the one that was already
       * allocated.
       */
        os_free_aligned(_util_sparse_array_node_data(node));
        반환 prev_node;
    } 아니면 {
        반환 node;
    }
}

void * util_sparse_array_get(util_sparse_array * arr, uint64_t idx) {
    const unsigned node_size_log2 = arr->node_size_log2;
    uintptr_t      root           = p_atomic_read(&arr->root);
    만약 (unlikely(!root)) {
        unsigned root_level = 0;
        uint64_t idx_iter   = idx >> node_size_log2;
        동안 (idx_iter) {
            idx_iter >>= node_size_log2;
            root_level++;
        }
        uintptr_t new_root = _util_sparse_array_node_alloc(arr, root_level);
        root               = _util_sparse_array_set_or_free_node(&arr->root, NULL_NODE, new_root);
    }

    동안 (1) {
        unsigned root_level = _util_sparse_array_node_level(root);
        uint64_t root_idx   = idx >> (root_level * node_size_log2);
        만약 (likely(root_idx < (1ull << node_size_log2))) {
            break;
        }

        /* In this case, we have a root but its level is low enough that the
       * requested index is out-of-bounds.
       */
        uintptr_t new_root = _util_sparse_array_node_alloc(arr, root_level + 1);

        uintptr_t * new_root_children = (uintptr_t *) _util_sparse_array_node_data(new_root);
        new_root_children[0]          = root;

        /* We only add one at a time instead of the whole tree because it's
       * easier to ensure correctness of both the tree building and the
       * clean-up path.  Because we're only adding one node we never have to
       * worry about trying to free multiple things without freeing the old
       * things.
       */
        root = _util_sparse_array_set_or_free_node(&arr->root, root, new_root);
    }

    void *   node_data  = _util_sparse_array_node_data(root);
    unsigned node_level = _util_sparse_array_node_level(root);
    동안 (node_level > 0) {
        uint64_t child_idx = (idx >> (node_level * node_size_log2)) & ((1ull << node_size_log2) - 1);

        uintptr_t * children = (uintptr_t *) node_data;
        uintptr_t   child    = p_atomic_read(&children[child_idx]);

        만약 (unlikely(!child)) {
            child = _util_sparse_array_node_alloc(arr, node_level - 1);
            child = _util_sparse_array_set_or_free_node(&children[child_idx], NULL_NODE, child);
        }

        node_data  = _util_sparse_array_node_data(child);
        node_level = _util_sparse_array_node_level(child);
    }

    uint64_t elem_idx = idx & ((1ull << node_size_log2) - 1);
    반환 (void *) ((char *) node_data + (elem_idx * arr->elem_size));
}
