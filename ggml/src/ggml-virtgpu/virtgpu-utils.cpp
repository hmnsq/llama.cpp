#ضمن "virtgpu-utils.h"

#ضمن <malloc.h>
#ضمن <stdlib.h>

#ضمن <cstring>

#وصف_ڪريو ڳنڍ_ALLOC_ALIGN 64
#وصف_ڪريو ڳنڍ_PTR_نقاب    (~((uintptr_t) ڳنڍ_ALLOC_ALIGN - 1))
#وصف_ڪريو ڳنڍ_LEVEL_نقاب  ((uintptr_t) ڳنڍ_ALLOC_ALIGN - 1)
#وصف_ڪريو NULL_ڳنڍ        0

#وصف_ڪريو os_malloc_aligned(سائيز_2, _align) _aligned_malloc(سائيز_2, _align)
#وصف_ڪريو os_free_aligned(_ptr)            حرر(_ptr)
#وصف_ڪريو p_atomic_cmpxchg(v, old, _new)   __sync_val_compare_and_swap((v), (old), (_new))

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
#جيڪڏهن_پورو
}

فراغ util_sparse_array_init(util_sparse_array * arr, size_t elem_سائيز, size_t ڳنڍ_سائيز) {
    memset(arr, 0, حجم(*arr));
    arr->elem_سائيز      = elem_سائيز;
    arr->ڳنڍ_سائيز_log2 = util_logbase2_64(ڳنڍ_سائيز);
    أكد(ڳنڍ_سائيز >= 2 && ڳنڍ_سائيز == (1ull << arr->ڳنڍ_سائيز_log2));
}

ساكن مضمن فراغ * os_malloc_aligned(size_t size, size_t alignment) {
    فراغ * ptr;
    alignment = (alignment + حجم(فراغ *) - 1) & ~(حجم(فراغ *) - 1);
    إذا (posix_memalign(&ptr, alignment, size) != 0) {
        أرجع عدم;
    }
    أرجع ptr;
}

ساكن مضمن فراغ * util_sparse_array_ڳنڍ_معلومات(uintptr_t handle) {
    أرجع (فراغ *) (handle & ڳنڍ_PTR_نقاب);
}

ساكن مضمن بنانشان util_sparse_array_ڳنڍ_level(uintptr_t handle) {
    أرجع handle & ڳنڍ_LEVEL_نقاب;
}

ساكن مضمن فراغ util_sparse_array_ڳنڍ_پڄاڻي(util_sparse_array * arr, uintptr_t ڳنڍ) {
    إذا (util_sparse_array_ڳنڍ_level(ڳنڍ) > 0) {
        uintptr_t * children  = (uintptr_t *) util_sparse_array_ڳنڍ_معلومات(ڳنڍ);
        size_t      ڳنڍ_سائيز = 1ull << arr->ڳنڍ_سائيز_log2;
        لكل (size_t i = 0; i < ڳنڍ_سائيز; i++) {
            إذا (children[i]) {
                util_sparse_array_ڳنڍ_پڄاڻي(arr, children[i]);
            }
        }
    }

    os_free_aligned(util_sparse_array_ڳنڍ_معلومات(ڳنڍ));
}

ساكن مضمن uintptr_t util_sparse_array_ڳنڍ(فراغ * معلومات, بنانشان level) {
    أكد(معلومات != عدم);
    أكد(((uintptr_t) معلومات & ڳنڍ_LEVEL_نقاب) == 0);
    أكد((level & ڳنڍ_PTR_نقاب) == 0);
    أرجع (uintptr_t) معلومات | level;
}

مضمن uintptr_t util_sparse_array_ڳنڍ_alloc(util_sparse_array * arr, بنانشان level) {
    size_t size;
    إذا (level == 0) {
        size = arr->elem_سائيز << arr->ڳنڍ_سائيز_log2;
    } وإلا {
        size = حجم(uintptr_t) << arr->ڳنڍ_سائيز_log2;
    }

    فراغ * معلومات = os_malloc_aligned(size, ڳنڍ_ALLOC_ALIGN);
    memset(معلومات, 0, size);

    أرجع util_sparse_array_ڳنڍ(معلومات, level);
}

ساكن مضمن uintptr_t util_sparse_array_سيٽ_or_free_ڳنڍ(uintptr_t * ڳنڍ_ptr, uintptr_t cmp_ڳنڍ, uintptr_t ڳنڍ) {
    uintptr_t prev_ڳنڍ = p_atomic_cmpxchg(ڳنڍ_ptr, cmp_ڳنڍ, ڳنڍ);

    إذا (prev_ڳنڍ != cmp_ڳنڍ) {
        /* We lost the race.  Free this one and return the one that was already
       * allocated.
       */
        os_free_aligned(util_sparse_array_ڳنڍ_معلومات(ڳنڍ));
        أرجع prev_ڳنڍ;
    } وإلا {
        أرجع ڳنڍ;
    }
}

فراغ * util_sparse_array_get(util_sparse_array * arr, uint64_t idx) {
    ثابت بنانشان ڳنڍ_سائيز_log2 = arr->ڳنڍ_سائيز_log2;
    uintptr_t      پاڙ           = p_atomic_read(&arr->پاڙ);
    إذا (unlikely(!پاڙ)) {
        بنانشان پاڙ_level = 0;
        uint64_t idx_iter   = idx >> ڳنڍ_سائيز_log2;
        طالما (idx_iter) {
            idx_iter >>= ڳنڍ_سائيز_log2;
            پاڙ_level++;
        }
        uintptr_t new_پاڙ = util_sparse_array_ڳنڍ_alloc(arr, پاڙ_level);
        پاڙ               = util_sparse_array_سيٽ_or_free_ڳنڍ(&arr->پاڙ, NULL_ڳنڍ, new_پاڙ);
    }

    طالما (1) {
        بنانشان پاڙ_level = util_sparse_array_ڳنڍ_level(پاڙ);
        uint64_t پاڙ_idx   = idx >> (پاڙ_level * ڳنڍ_سائيز_log2);
        إذا (likely(پاڙ_idx < (1ull << ڳنڍ_سائيز_log2))) {
            اكسر;
        }

        /* In this case, we have a root but its level is low enough that the
       * requested index is out-of-bounds.
       */
        uintptr_t new_پاڙ = util_sparse_array_ڳنڍ_alloc(arr, پاڙ_level + 1);

        uintptr_t * new_پاڙ_children = (uintptr_t *) util_sparse_array_ڳنڍ_معلومات(new_پاڙ);
        new_پاڙ_children[0]          = پاڙ;

        /* We only add one at a time instead of the whole tree because it's
       * easier to ensure correctness of both the tree building and the
       * clean-up path.  Because we're only adding one node we never have to
       * worry about trying to free multiple things without freeing the old
       * things.
       */
        پاڙ = util_sparse_array_سيٽ_or_free_ڳنڍ(&arr->پاڙ, پاڙ, new_پاڙ);
    }

    فراغ *   ڳنڍ_معلومات  = util_sparse_array_ڳنڍ_معلومات(پاڙ);
    بنانشان ڳنڍ_level = util_sparse_array_ڳنڍ_level(پاڙ);
    طالما (ڳنڍ_level > 0) {
        uint64_t ٻار_idx = (idx >> (ڳنڍ_level * ڳنڍ_سائيز_log2)) & ((1ull << ڳنڍ_سائيز_log2) - 1);

        uintptr_t * children = (uintptr_t *) ڳنڍ_معلومات;
        uintptr_t   ٻار    = p_atomic_read(&children[ٻار_idx]);

        إذا (unlikely(!ٻار)) {
            ٻار = util_sparse_array_ڳنڍ_alloc(arr, ڳنڍ_level - 1);
            ٻار = util_sparse_array_سيٽ_or_free_ڳنڍ(&children[ٻار_idx], NULL_ڳنڍ, ٻار);
        }

        ڳنڍ_معلومات  = util_sparse_array_ڳنڍ_معلومات(ٻار);
        ڳنڍ_level = util_sparse_array_ڳنڍ_level(ٻار);
    }

    uint64_t elem_idx = idx & ((1ull << ڳنڍ_سائيز_log2) - 1);
    أرجع (فراغ *) ((حرف *) ڳنڍ_معلومات + (elem_idx * arr->elem_سائيز));
}
