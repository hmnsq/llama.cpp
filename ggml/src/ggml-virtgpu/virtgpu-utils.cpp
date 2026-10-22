#ضمن "virtgpu-utils.h"

#ضمن <malloc.h>
#ضمن <stdlib.h>

#ضمن <cstring>

#تعریف_کړه غوټه_ALLOC_ALIGN 64
#تعریف_کړه غوټه_PTR_ماسک    (~((uintptr_t) غوټه_ALLOC_ALIGN - 1))
#تعریف_کړه غوټه_LEVEL_ماسک  ((uintptr_t) غوټه_ALLOC_ALIGN - 1)
#تعریف_کړه NULL_غوټه        0

#تعریف_کړه os_malloc_aligned(اندازه_2, _align) _aligned_malloc(اندازه_2, _align)
#تعریف_کړه os_free_aligned(_ptr)            حرر(_ptr)
#تعریف_کړه p_atomic_cmpxchg(v, old, _new)   __sync_val_compare_and_swap((v), (old), (_new))

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
#د_که_پای
}

فراغ util_sparse_array_init(util_sparse_array * arr, size_t elem_اندازه, size_t غوټه_اندازه) {
    memset(arr, 0, حجم(*arr));
    arr->elem_اندازه      = elem_اندازه;
    arr->غوټه_اندازه_log2 = util_logbase2_64(غوټه_اندازه);
    أكد(غوټه_اندازه >= 2 && غوټه_اندازه == (1ull << arr->غوټه_اندازه_log2));
}

ساكن مضمن فراغ * os_malloc_aligned(size_t size, size_t alignment) {
    فراغ * ptr;
    alignment = (alignment + حجم(فراغ *) - 1) & ~(حجم(فراغ *) - 1);
    إذا (posix_memalign(&ptr, alignment, size) != 0) {
        أرجع عدم;
    }
    أرجع ptr;
}

ساكن مضمن فراغ * util_sparse_array_غوټه_معلومات(uintptr_t handle) {
    أرجع (فراغ *) (handle & غوټه_PTR_ماسک);
}

ساكن مضمن بینښه util_sparse_array_غوټه_level(uintptr_t handle) {
    أرجع handle & غوټه_LEVEL_ماسک;
}

ساكن مضمن فراغ util_sparse_array_غوټه_پای(util_sparse_array * arr, uintptr_t غوټه) {
    إذا (util_sparse_array_غوټه_level(غوټه) > 0) {
        uintptr_t * children  = (uintptr_t *) util_sparse_array_غوټه_معلومات(غوټه);
        size_t      غوټه_اندازه = 1ull << arr->غوټه_اندازه_log2;
        لكل (size_t i = 0; i < غوټه_اندازه; i++) {
            إذا (children[i]) {
                util_sparse_array_غوټه_پای(arr, children[i]);
            }
        }
    }

    os_free_aligned(util_sparse_array_غوټه_معلومات(غوټه));
}

ساكن مضمن uintptr_t util_sparse_array_غوټه(فراغ * معلومات, بینښه level) {
    أكد(معلومات != عدم);
    أكد(((uintptr_t) معلومات & غوټه_LEVEL_ماسک) == 0);
    أكد((level & غوټه_PTR_ماسک) == 0);
    أرجع (uintptr_t) معلومات | level;
}

مضمن uintptr_t util_sparse_array_غوټه_alloc(util_sparse_array * arr, بینښه level) {
    size_t size;
    إذا (level == 0) {
        size = arr->elem_اندازه << arr->غوټه_اندازه_log2;
    } وإلا {
        size = حجم(uintptr_t) << arr->غوټه_اندازه_log2;
    }

    فراغ * معلومات = os_malloc_aligned(size, غوټه_ALLOC_ALIGN);
    memset(معلومات, 0, size);

    أرجع util_sparse_array_غوټه(معلومات, level);
}

ساكن مضمن uintptr_t util_sparse_array_سېټ_or_free_غوټه(uintptr_t * غوټه_ptr, uintptr_t cmp_غوټه, uintptr_t غوټه) {
    uintptr_t prev_غوټه = p_atomic_cmpxchg(غوټه_ptr, cmp_غوټه, غوټه);

    إذا (prev_غوټه != cmp_غوټه) {
        /* We lost the race.  Free this one and return the one that was already
       * allocated.
       */
        os_free_aligned(util_sparse_array_غوټه_معلومات(غوټه));
        أرجع prev_غوټه;
    } وإلا {
        أرجع غوټه;
    }
}

فراغ * util_sparse_array_get(util_sparse_array * arr, uint64_t idx) {
    ثابت بینښه غوټه_اندازه_log2 = arr->غوټه_اندازه_log2;
    uintptr_t      ریښه           = p_atomic_read(&arr->ریښه);
    إذا (unlikely(!ریښه)) {
        بینښه ریښه_level = 0;
        uint64_t idx_iter   = idx >> غوټه_اندازه_log2;
        طالما (idx_iter) {
            idx_iter >>= غوټه_اندازه_log2;
            ریښه_level++;
        }
        uintptr_t new_ریښه = util_sparse_array_غوټه_alloc(arr, ریښه_level);
        ریښه               = util_sparse_array_سېټ_or_free_غوټه(&arr->ریښه, NULL_غوټه, new_ریښه);
    }

    طالما (1) {
        بینښه ریښه_level = util_sparse_array_غوټه_level(ریښه);
        uint64_t ریښه_idx   = idx >> (ریښه_level * غوټه_اندازه_log2);
        إذا (likely(ریښه_idx < (1ull << غوټه_اندازه_log2))) {
            اكسر;
        }

        /* In this case, we have a root but its level is low enough that the
       * requested index is out-of-bounds.
       */
        uintptr_t new_ریښه = util_sparse_array_غوټه_alloc(arr, ریښه_level + 1);

        uintptr_t * new_ریښه_children = (uintptr_t *) util_sparse_array_غوټه_معلومات(new_ریښه);
        new_ریښه_children[0]          = ریښه;

        /* We only add one at a time instead of the whole tree because it's
       * easier to ensure correctness of both the tree building and the
       * clean-up path.  Because we're only adding one node we never have to
       * worry about trying to free multiple things without freeing the old
       * things.
       */
        ریښه = util_sparse_array_سېټ_or_free_غوټه(&arr->ریښه, ریښه, new_ریښه);
    }

    فراغ *   غوټه_معلومات  = util_sparse_array_غوټه_معلومات(ریښه);
    بینښه غوټه_level = util_sparse_array_غوټه_level(ریښه);
    طالما (غوټه_level > 0) {
        uint64_t ماشوم_idx = (idx >> (غوټه_level * غوټه_اندازه_log2)) & ((1ull << غوټه_اندازه_log2) - 1);

        uintptr_t * children = (uintptr_t *) غوټه_معلومات;
        uintptr_t   ماشوم    = p_atomic_read(&children[ماشوم_idx]);

        إذا (unlikely(!ماشوم)) {
            ماشوم = util_sparse_array_غوټه_alloc(arr, غوټه_level - 1);
            ماشوم = util_sparse_array_سېټ_or_free_غوټه(&children[ماشوم_idx], NULL_غوټه, ماشوم);
        }

        غوټه_معلومات  = util_sparse_array_غوټه_معلومات(ماشوم);
        غوټه_level = util_sparse_array_غوټه_level(ماشوم);
    }

    uint64_t elem_idx = idx & ((1ull << غوټه_اندازه_log2) - 1);
    أرجع (فراغ *) ((حرف *) غوټه_معلومات + (elem_idx * arr->elem_اندازه));
}
