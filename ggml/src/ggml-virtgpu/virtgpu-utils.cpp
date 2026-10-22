#ضمن "virtgpu-utils.h"

#ضمن <malloc.h>
#ضمن <stdlib.h>

#ضمن <cstring>

#define كلي_ALLOC_ALIGN 64
#define كلي_PTR_MASK    (~((uintptr_t) كلي_ALLOC_ALIGN - 1))
#define كلي_LEVEL_MASK  ((uintptr_t) كلي_ALLOC_ALIGN - 1)
#define NULL_كلي        0

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

فراغ util_sparse_array_init(util_sparse_array * arr, size_t elem_size, size_t كلي_size) {
    memset(arr, 0, حجم(*arr));
    arr->elem_size      = elem_size;
    arr->كلي_size_log2 = util_logbase2_64(كلي_size);
    أكد(كلي_size >= 2 && كلي_size == (1ull << arr->كلي_size_log2));
}

ساكن مضمن فراغ * os_malloc_aligned(size_t size, size_t alignment) {
    فراغ * ptr;
    alignment = (alignment + حجم(فراغ *) - 1) & ~(حجم(فراغ *) - 1);
    إذا (posix_memalign(&ptr, alignment, size) != 0) {
        أرجع عدم;
    }
    أرجع ptr;
}

ساكن مضمن فراغ * util_sparse_array_كلي_data(uintptr_t handle) {
    أرجع (فراغ *) (handle & كلي_PTR_MASK);
}

ساكن مضمن unsigned util_sparse_array_كلي_level(uintptr_t handle) {
    أرجع handle & كلي_LEVEL_MASK;
}

ساكن مضمن فراغ util_sparse_array_كلي_finish(util_sparse_array * arr, uintptr_t كلي) {
    إذا (util_sparse_array_كلي_level(كلي) > 0) {
        uintptr_t * children  = (uintptr_t *) util_sparse_array_كلي_data(كلي);
        size_t      كلي_size = 1ull << arr->كلي_size_log2;
        لكل (size_t i = 0; i < كلي_size; i++) {
            إذا (children[i]) {
                util_sparse_array_كلي_finish(arr, children[i]);
            }
        }
    }

    os_free_aligned(util_sparse_array_كلي_data(كلي));
}

ساكن مضمن uintptr_t util_sparse_array_كلي(فراغ * data, unsigned level) {
    أكد(data != عدم);
    أكد(((uintptr_t) data & كلي_LEVEL_MASK) == 0);
    أكد((level & كلي_PTR_MASK) == 0);
    أرجع (uintptr_t) data | level;
}

مضمن uintptr_t util_sparse_array_كلي_alloc(util_sparse_array * arr, unsigned level) {
    size_t size;
    إذا (level == 0) {
        size = arr->elem_size << arr->كلي_size_log2;
    } وإلا {
        size = حجم(uintptr_t) << arr->كلي_size_log2;
    }

    فراغ * data = os_malloc_aligned(size, كلي_ALLOC_ALIGN);
    memset(data, 0, size);

    أرجع util_sparse_array_كلي(data, level);
}

ساكن مضمن uintptr_t util_sparse_array_set_or_free_كلي(uintptr_t * كلي_ptr, uintptr_t cmp_كلي, uintptr_t كلي) {
    uintptr_t prev_كلي = p_atomic_cmpxchg(كلي_ptr, cmp_كلي, كلي);

    إذا (prev_كلي != cmp_كلي) {
        /* We lost the race.  Free this one and return the one that was already
       * allocated.
       */
        os_free_aligned(util_sparse_array_كلي_data(كلي));
        أرجع prev_كلي;
    } وإلا {
        أرجع كلي;
    }
}

فراغ * util_sparse_array_get(util_sparse_array * arr, uint64_t idx) {
    ثابت unsigned كلي_size_log2 = arr->كلي_size_log2;
    uintptr_t      سيو           = p_atomic_read(&arr->سيو);
    إذا (unlikely(!سيو)) {
        unsigned سيو_level = 0;
        uint64_t idx_iter   = idx >> كلي_size_log2;
        طالما (idx_iter) {
            idx_iter >>= كلي_size_log2;
            سيو_level++;
        }
        uintptr_t new_سيو = util_sparse_array_كلي_alloc(arr, سيو_level);
        سيو               = util_sparse_array_set_or_free_كلي(&arr->سيو, NULL_كلي, new_سيو);
    }

    طالما (1) {
        unsigned سيو_level = util_sparse_array_كلي_level(سيو);
        uint64_t سيو_idx   = idx >> (سيو_level * كلي_size_log2);
        إذا (likely(سيو_idx < (1ull << كلي_size_log2))) {
            اكسر;
        }

        /* In this case, we have a root but its level is low enough that the
       * requested index is out-of-bounds.
       */
        uintptr_t new_سيو = util_sparse_array_كلي_alloc(arr, سيو_level + 1);

        uintptr_t * new_سيو_children = (uintptr_t *) util_sparse_array_كلي_data(new_سيو);
        new_سيو_children[0]          = سيو;

        /* We only add one at a time instead of the whole tree because it's
       * easier to ensure correctness of both the tree building and the
       * clean-up path.  Because we're only adding one node we never have to
       * worry about trying to free multiple things without freeing the old
       * things.
       */
        سيو = util_sparse_array_set_or_free_كلي(&arr->سيو, سيو, new_سيو);
    }

    فراغ *   كلي_data  = util_sparse_array_كلي_data(سيو);
    unsigned كلي_level = util_sparse_array_كلي_level(سيو);
    طالما (كلي_level > 0) {
        uint64_t child_idx = (idx >> (كلي_level * كلي_size_log2)) & ((1ull << كلي_size_log2) - 1);

        uintptr_t * children = (uintptr_t *) كلي_data;
        uintptr_t   child    = p_atomic_read(&children[child_idx]);

        إذا (unlikely(!child)) {
            child = util_sparse_array_كلي_alloc(arr, كلي_level - 1);
            child = util_sparse_array_set_or_free_كلي(&children[child_idx], NULL_كلي, child);
        }

        كلي_data  = util_sparse_array_كلي_data(child);
        كلي_level = util_sparse_array_كلي_level(child);
    }

    uint64_t elem_idx = idx & ((1ull << كلي_size_log2) - 1);
    أرجع (فراغ *) ((حرف *) كلي_data + (elem_idx * arr->elem_size));
}
