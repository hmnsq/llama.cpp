#अन्तर्भाव "virtgpu-utils.h"

#अन्तर्भाव <malloc.h>
#अन्तर्भाव <stdlib.h>

#अन्तर्भाव <cstring>

#define ग्रन्थिः_ALLOC_ALIGN 64
#define ग्रन्थिः_PTR_MASK    (~((uintptr_t) ग्रन्थिः_ALLOC_ALIGN - 1))
#define ग्रन्थिः_LEVEL_MASK  ((uintptr_t) ग्रन्थिः_ALLOC_ALIGN - 1)
#define NULL_ग्रन्थिः        0

#define os_malloc_aligned(_size, _align) _aligned_malloc(_size, _align)
#define os_free_aligned(_ptr)            मुक्त(_ptr)
#define p_atomic_cmpxchg(v, old, _new)   __sync_val_compare_and_swap((v), (old), (_new))

स्थिर अन्तःस्थ uint64_t util_logbase2_64(uint64_t n) {
#सचे defined(HAVE___BUILTIN_CLZLL)
    पटिनिवत्त ((परिमाण(uint64_t) * 8 - 1) - __builtin_clzll(n | 1));
#अञ्ञथा
    uint64_t pos = 0ull;
    सचे (n >= 1ull << 32) {
        n >>= 32;
        pos += 32;
    }
    सचे (n >= 1ull << 16) {
        n >>= 16;
        pos += 16;
    }
    सचे (n >= 1ull << 8) {
        n >>= 8;
        pos += 8;
    }
    सचे (n >= 1ull << 4) {
        n >>= 4;
        pos += 4;
    }
    सचे (n >= 1ull << 2) {
        n >>= 2;
        pos += 2;
    }
    सचे (n >= 1ull << 1) {
        pos += 1;
    }
    पटिनिवत्त pos;
#endif
}

शून्य util_sparse_array_init(util_sparse_array * arr, size_t elem_size, size_t ग्रन्थिः_size) {
    memset(arr, 0, परिमाण(*arr));
    arr->elem_size      = elem_size;
    arr->ग्रन्थिः_size_log2 = util_logbase2_64(ग्रन्थिः_size);
    प्रतिज्ञा(ग्रन्थिः_size >= 2 && ग्रन्थिः_size == (1ull << arr->ग्रन्थिः_size_log2));
}

स्थिर अन्तःस्थ शून्य * os_malloc_aligned(size_t size, size_t alignment) {
    शून्य * ptr;
    alignment = (alignment + परिमाण(शून्य *) - 1) & ~(परिमाण(शून्य *) - 1);
    सचे (posix_memalign(&ptr, alignment, size) != 0) {
        पटिनिवत्त रिक्त;
    }
    पटिनिवत्त ptr;
}

स्थिर अन्तःस्थ शून्य * util_sparse_array_ग्रन्थिः_data(uintptr_t handle) {
    पटिनिवत्त (शून्य *) (handle & ग्रन्थिः_PTR_MASK);
}

स्थिर अन्तःस्थ unsigned util_sparse_array_ग्रन्थिः_level(uintptr_t handle) {
    पटिनिवत्त handle & ग्रन्थिः_LEVEL_MASK;
}

स्थिर अन्तःस्थ शून्य util_sparse_array_ग्रन्थिः_finish(util_sparse_array * arr, uintptr_t ग्रन्थिः) {
    सचे (util_sparse_array_ग्रन्थिः_level(ग्रन्थिः) > 0) {
        uintptr_t * children  = (uintptr_t *) util_sparse_array_ग्रन्थिः_data(ग्रन्थिः);
        size_t      ग्रन्थिः_size = 1ull << arr->ग्रन्थिः_size_log2;
        अत्थाय (size_t i = 0; i < ग्रन्थिः_size; i++) {
            सचे (children[i]) {
                util_sparse_array_ग्रन्थिः_finish(arr, children[i]);
            }
        }
    }

    os_free_aligned(util_sparse_array_ग्रन्थिः_data(ग्रन्थिः));
}

स्थिर अन्तःस्थ uintptr_t util_sparse_array_ग्रन्थिः(शून्य * data, unsigned level) {
    प्रतिज्ञा(data != रिक्त);
    प्रतिज्ञा(((uintptr_t) data & ग्रन्थिः_LEVEL_MASK) == 0);
    प्रतिज्ञा((level & ग्रन्थिः_PTR_MASK) == 0);
    पटिनिवत्त (uintptr_t) data | level;
}

अन्तःस्थ uintptr_t util_sparse_array_ग्रन्थिः_alloc(util_sparse_array * arr, unsigned level) {
    size_t size;
    सचे (level == 0) {
        size = arr->elem_size << arr->ग्रन्थिः_size_log2;
    } अञ्ञथा {
        size = परिमाण(uintptr_t) << arr->ग्रन्थिः_size_log2;
    }

    शून्य * data = os_malloc_aligned(size, ग्रन्थिः_ALLOC_ALIGN);
    memset(data, 0, size);

    पटिनिवत्त util_sparse_array_ग्रन्थिः(data, level);
}

स्थिर अन्तःस्थ uintptr_t util_sparse_array_set_or_free_ग्रन्थिः(uintptr_t * ग्रन्थिः_ptr, uintptr_t cmp_ग्रन्थिः, uintptr_t ग्रन्थिः) {
    uintptr_t prev_ग्रन्थिः = p_atomic_cmpxchg(ग्रन्थिः_ptr, cmp_ग्रन्थिः, ग्रन्थिः);

    सचे (prev_ग्रन्थिः != cmp_ग्रन्थिः) {
        /* We lost the race.  Free this one and return the one that was already
       * allocated.
       */
        os_free_aligned(util_sparse_array_ग्रन्थिः_data(ग्रन्थिः));
        पटिनिवत्त prev_ग्रन्थिः;
    } अञ्ञथा {
        पटिनिवत्त ग्रन्थिः;
    }
}

शून्य * util_sparse_array_get(util_sparse_array * arr, uint64_t idx) {
    ध्रुव unsigned ग्रन्थिः_size_log2 = arr->ग्रन्थिः_size_log2;
    uintptr_t      मूलम्           = p_atomic_read(&arr->मूलम्);
    सचे (unlikely(!मूलम्)) {
        unsigned मूलम्_level = 0;
        uint64_t idx_iter   = idx >> ग्रन्थिः_size_log2;
        याव (idx_iter) {
            idx_iter >>= ग्रन्थिः_size_log2;
            मूलम्_level++;
        }
        uintptr_t new_मूलम् = util_sparse_array_ग्रन्थिः_alloc(arr, मूलम्_level);
        मूलम्               = util_sparse_array_set_or_free_ग्रन्थिः(&arr->मूलम्, NULL_ग्रन्थिः, new_मूलम्);
    }

    याव (1) {
        unsigned मूलम्_level = util_sparse_array_ग्रन्थिः_level(मूलम्);
        uint64_t मूलम्_idx   = idx >> (मूलम्_level * ग्रन्थिः_size_log2);
        सचे (likely(मूलम्_idx < (1ull << ग्रन्थिः_size_log2))) {
            भङ्ग;
        }

        /* In this case, we have a root but its level is low enough that the
       * requested index is out-of-bounds.
       */
        uintptr_t new_मूलम् = util_sparse_array_ग्रन्थिः_alloc(arr, मूलम्_level + 1);

        uintptr_t * new_मूलम्_children = (uintptr_t *) util_sparse_array_ग्रन्थिः_data(new_मूलम्);
        new_मूलम्_children[0]          = मूलम्;

        /* We only add one at a time instead of the whole tree because it's
       * easier to ensure correctness of both the tree building and the
       * clean-up path.  Because we're only adding one node we never have to
       * worry about trying to free multiple things without freeing the old
       * things.
       */
        मूलम् = util_sparse_array_set_or_free_ग्रन्थिः(&arr->मूलम्, मूलम्, new_मूलम्);
    }

    शून्य *   ग्रन्थिः_data  = util_sparse_array_ग्रन्थिः_data(मूलम्);
    unsigned ग्रन्थिः_level = util_sparse_array_ग्रन्थिः_level(मूलम्);
    याव (ग्रन्थिः_level > 0) {
        uint64_t child_idx = (idx >> (ग्रन्थिः_level * ग्रन्थिः_size_log2)) & ((1ull << ग्रन्थिः_size_log2) - 1);

        uintptr_t * children = (uintptr_t *) ग्रन्थिः_data;
        uintptr_t   child    = p_atomic_read(&children[child_idx]);

        सचे (unlikely(!child)) {
            child = util_sparse_array_ग्रन्थिः_alloc(arr, ग्रन्थिः_level - 1);
            child = util_sparse_array_set_or_free_ग्रन्थिः(&children[child_idx], NULL_ग्रन्थिः, child);
        }

        ग्रन्थिः_data  = util_sparse_array_ग्रन्थिः_data(child);
        ग्रन्थिः_level = util_sparse_array_ग्रन्थिः_level(child);
    }

    uint64_t elem_idx = idx & ((1ull << ग्रन्थिः_size_log2) - 1);
    पटिनिवत्त (शून्य *) ((वर्ण *) ग्रन्थिः_data + (elem_idx * arr->elem_size));
}
