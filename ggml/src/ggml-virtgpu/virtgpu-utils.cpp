#شامل "virtgpu-utils.h"

#شامل <malloc.h>
#شامل <stdlib.h>

#شامل <cstring>

#define گره_ALLOC_ALIGN 64
#define گره_PTR_MASK    (~((uintptr_t) گره_ALLOC_ALIGN - 1))
#define گره_LEVEL_MASK  ((uintptr_t) گره_ALLOC_ALIGN - 1)
#define NULL_گره        0

#define os_malloc_aligned(_size, _align) _aligned_malloc(_size, _align)
#define os_free_aligned(_ptr)            آزاد(_ptr)
#define p_atomic_cmpxchg(v, old, _new)   __sync_val_compare_and_swap((v), (old), (_new))

ایستا درونخطی uint64_t util_logbase2_64(uint64_t n) {
#اگر defined(HAVE___BUILTIN_CLZLL)
    بازگشت ((اندازه(uint64_t) * 8 - 1) - __builtin_clzll(n | 1));
#وگرنه
    uint64_t pos = 0ull;
    اگر (n >= 1ull << 32) {
        n >>= 32;
        pos += 32;
    }
    اگر (n >= 1ull << 16) {
        n >>= 16;
        pos += 16;
    }
    اگر (n >= 1ull << 8) {
        n >>= 8;
        pos += 8;
    }
    اگر (n >= 1ull << 4) {
        n >>= 4;
        pos += 4;
    }
    اگر (n >= 1ull << 2) {
        n >>= 2;
        pos += 2;
    }
    اگر (n >= 1ull << 1) {
        pos += 1;
    }
    بازگشت pos;
#endif
}

تهی util_sparse_array_init(util_sparse_array * arr, size_t elem_size, size_t گره_size) {
    memset(arr, 0, اندازه(*arr));
    arr->elem_size      = elem_size;
    arr->گره_size_log2 = util_logbase2_64(گره_size);
    أكد(گره_size >= 2 && گره_size == (1ull << arr->گره_size_log2));
}

ایستا درونخطی تهی * os_malloc_aligned(size_t size, size_t alignment) {
    تهی * ptr;
    alignment = (alignment + اندازه(تهی *) - 1) & ~(اندازه(تهی *) - 1);
    اگر (posix_memalign(&ptr, alignment, size) != 0) {
        بازگشت پوچ;
    }
    بازگشت ptr;
}

ایستا درونخطی تهی * util_sparse_array_گره_data(uintptr_t handle) {
    بازگشت (تهی *) (handle & گره_PTR_MASK);
}

ایستا درونخطی unsigned util_sparse_array_گره_level(uintptr_t handle) {
    بازگشت handle & گره_LEVEL_MASK;
}

ایستا درونخطی تهی util_sparse_array_گره_finish(util_sparse_array * arr, uintptr_t گره) {
    اگر (util_sparse_array_گره_level(گره) > 0) {
        uintptr_t * children  = (uintptr_t *) util_sparse_array_گره_data(گره);
        size_t      گره_size = 1ull << arr->گره_size_log2;
        برای (size_t i = 0; i < گره_size; i++) {
            اگر (children[i]) {
                util_sparse_array_گره_finish(arr, children[i]);
            }
        }
    }

    os_free_aligned(util_sparse_array_گره_data(گره));
}

ایستا درونخطی uintptr_t util_sparse_array_گره(تهی * data, unsigned level) {
    أكد(data != پوچ);
    أكد(((uintptr_t) data & گره_LEVEL_MASK) == 0);
    أكد((level & گره_PTR_MASK) == 0);
    بازگشت (uintptr_t) data | level;
}

درونخطی uintptr_t util_sparse_array_گره_alloc(util_sparse_array * arr, unsigned level) {
    size_t size;
    اگر (level == 0) {
        size = arr->elem_size << arr->گره_size_log2;
    } وگرنه {
        size = اندازه(uintptr_t) << arr->گره_size_log2;
    }

    تهی * data = os_malloc_aligned(size, گره_ALLOC_ALIGN);
    memset(data, 0, size);

    بازگشت util_sparse_array_گره(data, level);
}

ایستا درونخطی uintptr_t util_sparse_array_set_or_free_گره(uintptr_t * گره_ptr, uintptr_t cmp_گره, uintptr_t گره) {
    uintptr_t prev_گره = p_atomic_cmpxchg(گره_ptr, cmp_گره, گره);

    اگر (prev_گره != cmp_گره) {
        /* We lost the race.  Free this one and return the one that was already
       * allocated.
       */
        os_free_aligned(util_sparse_array_گره_data(گره));
        بازگشت prev_گره;
    } وگرنه {
        بازگشت گره;
    }
}

تهی * util_sparse_array_get(util_sparse_array * arr, uint64_t idx) {
    ثابت unsigned گره_size_log2 = arr->گره_size_log2;
    uintptr_t      ریشه           = p_atomic_read(&arr->ریشه);
    اگر (unlikely(!ریشه)) {
        unsigned ریشه_level = 0;
        uint64_t idx_iter   = idx >> گره_size_log2;
        وقتی (idx_iter) {
            idx_iter >>= گره_size_log2;
            ریشه_level++;
        }
        uintptr_t new_ریشه = util_sparse_array_گره_alloc(arr, ریشه_level);
        ریشه               = util_sparse_array_set_or_free_گره(&arr->ریشه, NULL_گره, new_ریشه);
    }

    وقتی (1) {
        unsigned ریشه_level = util_sparse_array_گره_level(ریشه);
        uint64_t ریشه_idx   = idx >> (ریشه_level * گره_size_log2);
        اگر (likely(ریشه_idx < (1ull << گره_size_log2))) {
            بشکن;
        }

        /* In this case, we have a root but its level is low enough that the
       * requested index is out-of-bounds.
       */
        uintptr_t new_ریشه = util_sparse_array_گره_alloc(arr, ریشه_level + 1);

        uintptr_t * new_ریشه_children = (uintptr_t *) util_sparse_array_گره_data(new_ریشه);
        new_ریشه_children[0]          = ریشه;

        /* We only add one at a time instead of the whole tree because it's
       * easier to ensure correctness of both the tree building and the
       * clean-up path.  Because we're only adding one node we never have to
       * worry about trying to free multiple things without freeing the old
       * things.
       */
        ریشه = util_sparse_array_set_or_free_گره(&arr->ریشه, ریشه, new_ریشه);
    }

    تهی *   گره_data  = util_sparse_array_گره_data(ریشه);
    unsigned گره_level = util_sparse_array_گره_level(ریشه);
    وقتی (گره_level > 0) {
        uint64_t child_idx = (idx >> (گره_level * گره_size_log2)) & ((1ull << گره_size_log2) - 1);

        uintptr_t * children = (uintptr_t *) گره_data;
        uintptr_t   child    = p_atomic_read(&children[child_idx]);

        اگر (unlikely(!child)) {
            child = util_sparse_array_گره_alloc(arr, گره_level - 1);
            child = util_sparse_array_set_or_free_گره(&children[child_idx], NULL_گره, child);
        }

        گره_data  = util_sparse_array_گره_data(child);
        گره_level = util_sparse_array_گره_level(child);
    }

    uint64_t elem_idx = idx & ((1ull << گره_size_log2) - 1);
    بازگشت (تهی *) ((نویسه *) گره_data + (elem_idx * arr->elem_size));
}
