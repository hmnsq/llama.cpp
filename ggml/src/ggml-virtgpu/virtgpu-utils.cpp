#ضمن "virtgpu-utils.h"

#ضمن <malloc.h>
#ضمن <stdlib.h>

#ضمن <cstring>

#define نود_ALLOC_ALIGN 64
#define نود_PTR_MASK    (~((uintptr_t) نود_ALLOC_ALIGN - 1))
#define نود_LEVEL_MASK  ((uintptr_t) نود_ALLOC_ALIGN - 1)
#define NULL_نود        0

#define os_malloc_aligned(_size, _align) _aligned_malloc(_size, _align)
#define os_free_aligned(_ptr)            حرر(_ptr)
#define p_atomic_cmpxchg(v, old, _new)   __sync_val_compare_and_swap((v), (old), (_new))

ساكن مضمن uint64_t util_logbase2_64(uint64_t n) {
#إذا defined(HAVE___BUILTIN_CLZLL)
    أرجع ((حجم(uint64_t) * 8 - 1) - __builtin_clzll(n | 1));
#وإلا
    uint64_t pos = 0ull;
    إذا (n >= 1ull << 32) {
        n >>= 32;
        pos += 32;
    }
    إذا (n >= 1ull << 16) {
        n >>= 16;
        pos += 16;
    }
    إذا (n >= 1ull << 8) {
        n >>= 8;
        pos += 8;
    }
    إذا (n >= 1ull << 4) {
        n >>= 4;
        pos += 4;
    }
    إذا (n >= 1ull << 2) {
        n >>= 2;
        pos += 2;
    }
    إذا (n >= 1ull << 1) {
        pos += 1;
    }
    أرجع pos;
#endif
}

فراغ util_sparse_array_init(util_sparse_array * arr, size_t elem_size, size_t نود_size) {
    memset(arr, 0, حجم(*arr));
    arr->elem_size      = elem_size;
    arr->نود_size_log2 = util_logbase2_64(نود_size);
    أكد(نود_size >= 2 && نود_size == (1ull << arr->نود_size_log2));
}

ساكن مضمن فراغ * os_malloc_aligned(size_t size, size_t alignment) {
    فراغ * ptr;
    alignment = (alignment + حجم(فراغ *) - 1) & ~(حجم(فراغ *) - 1);
    إذا (posix_memalign(&ptr, alignment, size) != 0) {
        أرجع عدم;
    }
    أرجع ptr;
}

ساكن مضمن فراغ * util_sparse_array_نود_data(uintptr_t handle) {
    أرجع (فراغ *) (handle & نود_PTR_MASK);
}

ساكن مضمن unsigned util_sparse_array_نود_level(uintptr_t handle) {
    أرجع handle & نود_LEVEL_MASK;
}

ساكن مضمن فراغ util_sparse_array_نود_finish(util_sparse_array * arr, uintptr_t نود) {
    إذا (util_sparse_array_نود_level(نود) > 0) {
        uintptr_t * children  = (uintptr_t *) util_sparse_array_نود_data(نود);
        size_t      نود_size = 1ull << arr->نود_size_log2;
        لكل (size_t i = 0; i < نود_size; i++) {
            إذا (children[i]) {
                util_sparse_array_نود_finish(arr, children[i]);
            }
        }
    }

    os_free_aligned(util_sparse_array_نود_data(نود));
}

ساكن مضمن uintptr_t util_sparse_array_نود(فراغ * data, unsigned level) {
    أكد(data != عدم);
    أكد(((uintptr_t) data & نود_LEVEL_MASK) == 0);
    أكد((level & نود_PTR_MASK) == 0);
    أرجع (uintptr_t) data | level;
}

مضمن uintptr_t util_sparse_array_نود_alloc(util_sparse_array * arr, unsigned level) {
    size_t size;
    إذا (level == 0) {
        size = arr->elem_size << arr->نود_size_log2;
    } وإلا {
        size = حجم(uintptr_t) << arr->نود_size_log2;
    }

    فراغ * data = os_malloc_aligned(size, نود_ALLOC_ALIGN);
    memset(data, 0, size);

    أرجع util_sparse_array_نود(data, level);
}

ساكن مضمن uintptr_t util_sparse_array_set_or_free_نود(uintptr_t * نود_ptr, uintptr_t cmp_نود, uintptr_t نود) {
    uintptr_t prev_نود = p_atomic_cmpxchg(نود_ptr, cmp_نود, نود);

    إذا (prev_نود != cmp_نود) {
        /* We lost the race.  Free this one and return the one that was already
       * allocated.
       */
        os_free_aligned(util_sparse_array_نود_data(نود));
        أرجع prev_نود;
    } وإلا {
        أرجع نود;
    }
}

فراغ * util_sparse_array_get(util_sparse_array * arr, uint64_t idx) {
    ثابت unsigned نود_size_log2 = arr->نود_size_log2;
    uintptr_t      اکر           = p_atomic_read(&arr->اکر);
    إذا (unlikely(!اکر)) {
        unsigned اکر_level = 0;
        uint64_t idx_iter   = idx >> نود_size_log2;
        طالما (idx_iter) {
            idx_iter >>= نود_size_log2;
            اکر_level++;
        }
        uintptr_t new_اکر = util_sparse_array_نود_alloc(arr, اکر_level);
        اکر               = util_sparse_array_set_or_free_نود(&arr->اکر, NULL_نود, new_اکر);
    }

    طالما (1) {
        unsigned اکر_level = util_sparse_array_نود_level(اکر);
        uint64_t اکر_idx   = idx >> (اکر_level * نود_size_log2);
        إذا (likely(اکر_idx < (1ull << نود_size_log2))) {
            اكسر;
        }

        /* In this case, we have a root but its level is low enough that the
       * requested index is out-of-bounds.
       */
        uintptr_t new_اکر = util_sparse_array_نود_alloc(arr, اکر_level + 1);

        uintptr_t * new_اکر_children = (uintptr_t *) util_sparse_array_نود_data(new_اکر);
        new_اکر_children[0]          = اکر;

        /* We only add one at a time instead of the whole tree because it's
       * easier to ensure correctness of both the tree building and the
       * clean-up path.  Because we're only adding one node we never have to
       * worry about trying to free multiple things without freeing the old
       * things.
       */
        اکر = util_sparse_array_set_or_free_نود(&arr->اکر, اکر, new_اکر);
    }

    فراغ *   نود_data  = util_sparse_array_نود_data(اکر);
    unsigned نود_level = util_sparse_array_نود_level(اکر);
    طالما (نود_level > 0) {
        uint64_t child_idx = (idx >> (نود_level * نود_size_log2)) & ((1ull << نود_size_log2) - 1);

        uintptr_t * children = (uintptr_t *) نود_data;
        uintptr_t   child    = p_atomic_read(&children[child_idx]);

        إذا (unlikely(!child)) {
            child = util_sparse_array_نود_alloc(arr, نود_level - 1);
            child = util_sparse_array_set_or_free_نود(&children[child_idx], NULL_نود, child);
        }

        نود_data  = util_sparse_array_نود_data(child);
        نود_level = util_sparse_array_نود_level(child);
    }

    uint64_t elem_idx = idx & ((1ull << نود_size_log2) - 1);
    أرجع (فراغ *) ((حرف *) نود_data + (elem_idx * arr->elem_size));
}
