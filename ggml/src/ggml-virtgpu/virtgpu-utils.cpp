#अन्तर्भाव "virtgpu-utils.h"

#अन्तर्भाव <malloc.h>
#अन्तर्भाव <stdlib.h>

#अन्तर्भाव <cstring>

#परिभाषय ग्रन्थिः_ALLOC_ALIGN 64
#परिभाषय ग्रन्थिः_PTR_आवरणम्    (~((uintptr_t) ग्रन्थिः_ALLOC_ALIGN - 1))
#परिभाषय ग्रन्थिः_LEVEL_आवरणम्  ((uintptr_t) ग्रन्थिः_ALLOC_ALIGN - 1)
#परिभाषय NULL_ग्रन्थिः        0

#परिभाषय os_malloc_aligned(आकार_2, _align) _aligned_malloc(आकार_2, _align)
#परिभाषय os_free_aligned(_ptr)            मुक्त(_ptr)
#परिभाषय p_atomic_cmpxchg(v, old, _new)   __sync_val_compare_and_swap((v), (old), (_new))

स्थिर अन्तःस्थ uint64_t util_logbase2_64(uint64_t n) {
#यदि defined(HAVE___BUILTIN_CLZLL)
    प्रत्यागम ((परिमाण(uint64_t) * 8 - 1) - __builtin_clzll(n | 1));
#अन्यथा
    uint64_t pos = 0ull;
    यदि (n >= 1ull << 32) {
        n >>= 32;
        pos += 32;
    }
    यदि (n >= 1ull << 16) {
        n >>= 16;
        pos += 16;
    }
    यदि (n >= 1ull << 8) {
        n >>= 8;
        pos += 8;
    }
    यदि (n >= 1ull << 4) {
        n >>= 4;
        pos += 4;
    }
    यदि (n >= 1ull << 2) {
        n >>= 2;
        pos += 2;
    }
    यदि (n >= 1ull << 1) {
        pos += 1;
    }
    प्रत्यागम pos;
#यदि_अन्तः
}

शून्य util_sparse_array_init(util_sparse_array * arr, size_t elem_आकार, size_t ग्रन्थिः_आकार) {
    स्मृतिं_पूरय(arr, 0, परिमाण(*arr));
    arr->elem_आकार      = elem_आकार;
    arr->ग्रन्थिः_आकार_log2 = util_logbase2_64(ग्रन्थिः_आकार);
    प्रतिज्ञा(ग्रन्थिः_आकार >= 2 && ग्रन्थिः_आकार == (1ull << arr->ग्रन्थिः_आकार_log2));
}

स्थिर अन्तःस्थ शून्य * os_malloc_aligned(size_t size, size_t alignment) {
    शून्य * ptr;
    alignment = (alignment + परिमाण(शून्य *) - 1) & ~(परिमाण(शून्य *) - 1);
    यदि (posix_memalign(&ptr, alignment, size) != 0) {
        प्रत्यागम रिक्त;
    }
    प्रत्यागम ptr;
}

स्थिर अन्तःस्थ शून्य * util_sparse_array_ग्रन्थिः_दत्त(uintptr_t handle) {
    प्रत्यागम (शून्य *) (handle & ग्रन्थिः_PTR_आवरणम्);
}

स्थिर अन्तःस्थ अचिह्नित util_sparse_array_ग्रन्थिः_level(uintptr_t handle) {
    प्रत्यागम handle & ग्रन्थिः_LEVEL_आवरणम्;
}

स्थिर अन्तःस्थ शून्य util_sparse_array_ग्रन्थिः_finish(util_sparse_array * arr, uintptr_t ग्रन्थिः) {
    यदि (util_sparse_array_ग्रन्थिः_level(ग्रन्थिः) > 0) {
        uintptr_t * children  = (uintptr_t *) util_sparse_array_ग्रन्थिः_दत्त(ग्रन्थिः);
        size_t      ग्रन्थिः_आकार = 1ull << arr->ग्रन्थिः_आकार_log2;
        प्रति (size_t i = 0; i < ग्रन्थिः_आकार; i++) {
            यदि (children[i]) {
                util_sparse_array_ग्रन्थिः_finish(arr, children[i]);
            }
        }
    }

    os_free_aligned(util_sparse_array_ग्रन्थिः_दत्त(ग्रन्थिः));
}

स्थिर अन्तःस्थ uintptr_t util_sparse_array_ग्रन्थिः(शून्य * दत्त, अचिह्नित level) {
    प्रतिज्ञा(दत्त != रिक्त);
    प्रतिज्ञा(((uintptr_t) दत्त & ग्रन्थिः_LEVEL_आवरणम्) == 0);
    प्रतिज्ञा((level & ग्रन्थिः_PTR_आवरणम्) == 0);
    प्रत्यागम (uintptr_t) दत्त | level;
}

अन्तःस्थ uintptr_t util_sparse_array_ग्रन्थिः_alloc(util_sparse_array * arr, अचिह्नित level) {
    size_t size;
    यदि (level == 0) {
        size = arr->elem_आकार << arr->ग्रन्थिः_आकार_log2;
    } अन्यथा {
        size = परिमाण(uintptr_t) << arr->ग्रन्थिः_आकार_log2;
    }

    शून्य * दत्त = os_malloc_aligned(size, ग्रन्थिः_ALLOC_ALIGN);
    स्मृतिं_पूरय(दत्त, 0, size);

    प्रत्यागम util_sparse_array_ग्रन्थिः(दत्त, level);
}

स्थिर अन्तःस्थ uintptr_t util_sparse_array_समुच्चयः_or_free_ग्रन्थिः(uintptr_t * ग्रन्थिः_ptr, uintptr_t cmp_ग्रन्थिः, uintptr_t ग्रन्थिः) {
    uintptr_t पूर्व_ग्रन्थिः = p_atomic_cmpxchg(ग्रन्थिः_ptr, cmp_ग्रन्थिः, ग्रन्थिः);

    यदि (पूर्व_ग्रन्थिः != cmp_ग्रन्थिः) {
        /* We lost the race.  Free this one and return the one that was already
       * allocated.
       */
        os_free_aligned(util_sparse_array_ग्रन्थिः_दत्त(ग्रन्थिः));
        प्रत्यागम पूर्व_ग्रन्थिः;
    } अन्यथा {
        प्रत्यागम ग्रन्थिः;
    }
}

शून्य * util_sparse_array_get(util_sparse_array * arr, uint64_t idx) {
    ध्रुव अचिह्नित ग्रन्थिः_आकार_log2 = arr->ग्रन्थिः_आकार_log2;
    uintptr_t      मूलम्           = p_atomic_read(&arr->मूलम्);
    यदि (unlikely(!मूलम्)) {
        अचिह्नित मूलम्_level = 0;
        uint64_t idx_iter   = idx >> ग्रन्थिः_आकार_log2;
        यावत् (idx_iter) {
            idx_iter >>= ग्रन्थिः_आकार_log2;
            मूलम्_level++;
        }
        uintptr_t new_मूलम् = util_sparse_array_ग्रन्थिः_alloc(arr, मूलम्_level);
        मूलम्               = util_sparse_array_समुच्चयः_or_free_ग्रन्थिः(&arr->मूलम्, NULL_ग्रन्थिः, new_मूलम्);
    }

    यावत् (1) {
        अचिह्नित मूलम्_level = util_sparse_array_ग्रन्थिः_level(मूलम्);
        uint64_t मूलम्_idx   = idx >> (मूलम्_level * ग्रन्थिः_आकार_log2);
        यदि (likely(मूलम्_idx < (1ull << ग्रन्थिः_आकार_log2))) {
            भङ्ग;
        }

        /* In this case, we have a root but its level is low enough that the
       * requested index is out-of-bounds.
       */
        uintptr_t new_मूलम् = util_sparse_array_ग्रन्थिः_alloc(arr, मूलम्_level + 1);

        uintptr_t * new_मूलम्_children = (uintptr_t *) util_sparse_array_ग्रन्थिः_दत्त(new_मूलम्);
        new_मूलम्_children[0]          = मूलम्;

        /* We only add one at a time instead of the whole tree because it's
       * easier to ensure correctness of both the tree building and the
       * clean-up path.  Because we're only adding one node we never have to
       * worry about trying to free multiple things without freeing the old
       * things.
       */
        मूलम् = util_sparse_array_समुच्चयः_or_free_ग्रन्थिः(&arr->मूलम्, मूलम्, new_मूलम्);
    }

    शून्य *   ग्रन्थिः_दत्त  = util_sparse_array_ग्रन्थिः_दत्त(मूलम्);
    अचिह्नित ग्रन्थिः_level = util_sparse_array_ग्रन्थिः_level(मूलम्);
    यावत् (ग्रन्थिः_level > 0) {
        uint64_t child_idx = (idx >> (ग्रन्थिः_level * ग्रन्थिः_आकार_log2)) & ((1ull << ग्रन्थिः_आकार_log2) - 1);

        uintptr_t * children = (uintptr_t *) ग्रन्थिः_दत्त;
        uintptr_t   child    = p_atomic_read(&children[child_idx]);

        यदि (unlikely(!child)) {
            child = util_sparse_array_ग्रन्थिः_alloc(arr, ग्रन्थिः_level - 1);
            child = util_sparse_array_समुच्चयः_or_free_ग्रन्थिः(&children[child_idx], NULL_ग्रन्थिः, child);
        }

        ग्रन्थिः_दत्त  = util_sparse_array_ग्रन्थिः_दत्त(child);
        ग्रन्थिः_level = util_sparse_array_ग्रन्थिः_level(child);
    }

    uint64_t elem_idx = idx & ((1ull << ग्रन्थिः_आकार_log2) - 1);
    प्रत्यागम (शून्य *) ((वर्ण *) ग्रन्थिः_दत्त + (elem_idx * arr->elem_आकार));
}
