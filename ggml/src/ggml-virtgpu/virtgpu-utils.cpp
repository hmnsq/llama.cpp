#شامل "virtgpu-utils.h"

#شامل <malloc.h>
#شامل <stdlib.h>

#شامل <cstring>

#تعریف_کرو گرہ_ALLOC_ALIGN 64
#تعریف_کرو گرہ_PTR_نقاب    (~((uintptr_t) گرہ_ALLOC_ALIGN - 1))
#تعریف_کرو گرہ_LEVEL_نقاب  ((uintptr_t) گرہ_ALLOC_ALIGN - 1)
#تعریف_کرو NULL_گرہ        0

#تعریف_کرو os_malloc_aligned(سائز_2, _align) _aligned_malloc(سائز_2, _align)
#تعریف_کرو os_free_aligned(_ptr)            آزاد(_ptr)
#تعریف_کرو p_atomic_cmpxchg(v, old, _new)   __sync_val_compare_and_swap((v), (old), (_new))

ساکن اندرونی uint64_t util_logbase2_64(uint64_t n) {
#اگر defined(HAVE___BUILTIN_CLZLL)
    واپس ((حجم(uint64_t) * 8 - 1) - __builtin_clzll(n | 1));
#ورنہ
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
    واپس pos;
#اگر_ختم
}

خالی util_sparse_array_init(util_sparse_array * arr, size_t elem_سائز, size_t گرہ_سائز) {
    memset(arr, 0, حجم(*arr));
    arr->elem_سائز      = elem_سائز;
    arr->گرہ_سائز_log2 = util_logbase2_64(گرہ_سائز);
    أكد(گرہ_سائز >= 2 && گرہ_سائز == (1ull << arr->گرہ_سائز_log2));
}

ساکن اندرونی خالی * os_malloc_aligned(size_t size, size_t alignment) {
    خالی * ptr;
    alignment = (alignment + حجم(خالی *) - 1) & ~(حجم(خالی *) - 1);
    اگر (posix_memalign(&ptr, alignment, size) != 0) {
        واپس تہی;
    }
    واپس ptr;
}

ساکن اندرونی خالی * util_sparse_array_گرہ_ڈیٹا(uintptr_t handle) {
    واپس (خالی *) (handle & گرہ_PTR_نقاب);
}

ساکن اندرونی غیرعلامتی util_sparse_array_گرہ_level(uintptr_t handle) {
    واپس handle & گرہ_LEVEL_نقاب;
}

ساکن اندرونی خالی util_sparse_array_گرہ_اختتام(util_sparse_array * arr, uintptr_t گرہ) {
    اگر (util_sparse_array_گرہ_level(گرہ) > 0) {
        uintptr_t * children  = (uintptr_t *) util_sparse_array_گرہ_ڈیٹا(گرہ);
        size_t      گرہ_سائز = 1ull << arr->گرہ_سائز_log2;
        برائے (size_t i = 0; i < گرہ_سائز; i++) {
            اگر (children[i]) {
                util_sparse_array_گرہ_اختتام(arr, children[i]);
            }
        }
    }

    os_free_aligned(util_sparse_array_گرہ_ڈیٹا(گرہ));
}

ساکن اندرونی uintptr_t util_sparse_array_گرہ(خالی * ڈیٹا, غیرعلامتی level) {
    أكد(ڈیٹا != تہی);
    أكد(((uintptr_t) ڈیٹا & گرہ_LEVEL_نقاب) == 0);
    أكد((level & گرہ_PTR_نقاب) == 0);
    واپس (uintptr_t) ڈیٹا | level;
}

اندرونی uintptr_t util_sparse_array_گرہ_alloc(util_sparse_array * arr, غیرعلامتی level) {
    size_t size;
    اگر (level == 0) {
        size = arr->elem_سائز << arr->گرہ_سائز_log2;
    } ورنہ {
        size = حجم(uintptr_t) << arr->گرہ_سائز_log2;
    }

    خالی * ڈیٹا = os_malloc_aligned(size, گرہ_ALLOC_ALIGN);
    memset(ڈیٹا, 0, size);

    واپس util_sparse_array_گرہ(ڈیٹا, level);
}

ساکن اندرونی uintptr_t util_sparse_array_سیٹ_or_free_گرہ(uintptr_t * گرہ_ptr, uintptr_t cmp_گرہ, uintptr_t گرہ) {
    uintptr_t پچھلا_گرہ = p_atomic_cmpxchg(گرہ_ptr, cmp_گرہ, گرہ);

    اگر (پچھلا_گرہ != cmp_گرہ) {
        /* We lost the race.  Free this one and return the one that was already
       * allocated.
       */
        os_free_aligned(util_sparse_array_گرہ_ڈیٹا(گرہ));
        واپس پچھلا_گرہ;
    } ورنہ {
        واپس گرہ;
    }
}

خالی * util_sparse_array_get(util_sparse_array * arr, uint64_t idx) {
    ثابت غیرعلامتی گرہ_سائز_log2 = arr->گرہ_سائز_log2;
    uintptr_t      جڑ           = p_atomic_read(&arr->جڑ);
    اگر (unlikely(!جڑ)) {
        غیرعلامتی جڑ_level = 0;
        uint64_t idx_iter   = idx >> گرہ_سائز_log2;
        جب (idx_iter) {
            idx_iter >>= گرہ_سائز_log2;
            جڑ_level++;
        }
        uintptr_t new_جڑ = util_sparse_array_گرہ_alloc(arr, جڑ_level);
        جڑ               = util_sparse_array_سیٹ_or_free_گرہ(&arr->جڑ, NULL_گرہ, new_جڑ);
    }

    جب (1) {
        غیرعلامتی جڑ_level = util_sparse_array_گرہ_level(جڑ);
        uint64_t جڑ_idx   = idx >> (جڑ_level * گرہ_سائز_log2);
        اگر (likely(جڑ_idx < (1ull << گرہ_سائز_log2))) {
            توڑو;
        }

        /* In this case, we have a root but its level is low enough that the
       * requested index is out-of-bounds.
       */
        uintptr_t new_جڑ = util_sparse_array_گرہ_alloc(arr, جڑ_level + 1);

        uintptr_t * new_جڑ_children = (uintptr_t *) util_sparse_array_گرہ_ڈیٹا(new_جڑ);
        new_جڑ_children[0]          = جڑ;

        /* We only add one at a time instead of the whole tree because it's
       * easier to ensure correctness of both the tree building and the
       * clean-up path.  Because we're only adding one node we never have to
       * worry about trying to free multiple things without freeing the old
       * things.
       */
        جڑ = util_sparse_array_سیٹ_or_free_گرہ(&arr->جڑ, جڑ, new_جڑ);
    }

    خالی *   گرہ_ڈیٹا  = util_sparse_array_گرہ_ڈیٹا(جڑ);
    غیرعلامتی گرہ_level = util_sparse_array_گرہ_level(جڑ);
    جب (گرہ_level > 0) {
        uint64_t بچہ_idx = (idx >> (گرہ_level * گرہ_سائز_log2)) & ((1ull << گرہ_سائز_log2) - 1);

        uintptr_t * children = (uintptr_t *) گرہ_ڈیٹا;
        uintptr_t   بچہ    = p_atomic_read(&children[بچہ_idx]);

        اگر (unlikely(!بچہ)) {
            بچہ = util_sparse_array_گرہ_alloc(arr, گرہ_level - 1);
            بچہ = util_sparse_array_سیٹ_or_free_گرہ(&children[بچہ_idx], NULL_گرہ, بچہ);
        }

        گرہ_ڈیٹا  = util_sparse_array_گرہ_ڈیٹا(بچہ);
        گرہ_level = util_sparse_array_گرہ_level(بچہ);
    }

    uint64_t elem_idx = idx & ((1ull << گرہ_سائز_log2) - 1);
    واپس (خالی *) ((حرف *) گرہ_ڈیٹا + (elem_idx * arr->elem_سائز));
}
