#ضمن "virtgpu-utils.h"

#ضمن <malloc.h>
#ضمن <stdlib.h>

#ضمن <cstring>

#تعریف_کرو گرہ_ALLOC_ALIGN 64
#تعریف_کرو گرہ_PTR_نقاب    (~((uintptr_t) گرہ_ALLOC_ALIGN - 1))
#تعریف_کرو گرہ_LEVEL_نقاب  ((uintptr_t) گرہ_ALLOC_ALIGN - 1)
#تعریف_کرو NULL_گرہ        0

#تعریف_کرو os_malloc_aligned(ناپ_2, _align) _aligned_malloc(ناپ_2, _align)
#تعریف_کرو os_free_aligned(_ptr)            حرر(_ptr)
#تعریف_کرو p_atomic_cmpxchg(v, old, _new)   __sync_val_compare_and_swap((v), (old), (_new))

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
#جے_انت
}

فراغ util_sparse_array_init(util_sparse_array * arr, size_t elem_ناپ, size_t گرہ_ناپ) {
    memset(arr, 0, حجم(*arr));
    arr->elem_ناپ      = elem_ناپ;
    arr->گرہ_ناپ_log2 = util_logbase2_64(گرہ_ناپ);
    أكد(گرہ_ناپ >= 2 && گرہ_ناپ == (1ull << arr->گرہ_ناپ_log2));
}

ساكن مضمن فراغ * os_malloc_aligned(size_t size, size_t alignment) {
    فراغ * ptr;
    alignment = (alignment + حجم(فراغ *) - 1) & ~(حجم(فراغ *) - 1);
    إذا (posix_memalign(&ptr, alignment, size) != 0) {
        أرجع عدم;
    }
    أرجع ptr;
}

ساكن مضمن فراغ * util_sparse_array_گرہ_انکڑے(uintptr_t handle) {
    أرجع (فراغ *) (handle & گرہ_PTR_نقاب);
}

ساكن مضمن بنانشانی util_sparse_array_گرہ_level(uintptr_t handle) {
    أرجع handle & گرہ_LEVEL_نقاب;
}

ساكن مضمن فراغ util_sparse_array_گرہ_انت(util_sparse_array * arr, uintptr_t گرہ) {
    إذا (util_sparse_array_گرہ_level(گرہ) > 0) {
        uintptr_t * children  = (uintptr_t *) util_sparse_array_گرہ_انکڑے(گرہ);
        size_t      گرہ_ناپ = 1ull << arr->گرہ_ناپ_log2;
        لكل (size_t i = 0; i < گرہ_ناپ; i++) {
            إذا (children[i]) {
                util_sparse_array_گرہ_انت(arr, children[i]);
            }
        }
    }

    os_free_aligned(util_sparse_array_گرہ_انکڑے(گرہ));
}

ساكن مضمن uintptr_t util_sparse_array_گرہ(فراغ * انکڑے, بنانشانی level) {
    أكد(انکڑے != عدم);
    أكد(((uintptr_t) انکڑے & گرہ_LEVEL_نقاب) == 0);
    أكد((level & گرہ_PTR_نقاب) == 0);
    أرجع (uintptr_t) انکڑے | level;
}

مضمن uintptr_t util_sparse_array_گرہ_alloc(util_sparse_array * arr, بنانشانی level) {
    size_t size;
    إذا (level == 0) {
        size = arr->elem_ناپ << arr->گرہ_ناپ_log2;
    } وإلا {
        size = حجم(uintptr_t) << arr->گرہ_ناپ_log2;
    }

    فراغ * انکڑے = os_malloc_aligned(size, گرہ_ALLOC_ALIGN);
    memset(انکڑے, 0, size);

    أرجع util_sparse_array_گرہ(انکڑے, level);
}

ساكن مضمن uintptr_t util_sparse_array_سیٹ_or_free_گرہ(uintptr_t * گرہ_ptr, uintptr_t cmp_گرہ, uintptr_t گرہ) {
    uintptr_t prev_گرہ = p_atomic_cmpxchg(گرہ_ptr, cmp_گرہ, گرہ);

    إذا (prev_گرہ != cmp_گرہ) {
        /* We lost the race.  Free this one and return the one that was already
       * allocated.
       */
        os_free_aligned(util_sparse_array_گرہ_انکڑے(گرہ));
        أرجع prev_گرہ;
    } وإلا {
        أرجع گرہ;
    }
}

فراغ * util_sparse_array_get(util_sparse_array * arr, uint64_t idx) {
    ثابت بنانشانی گرہ_ناپ_log2 = arr->گرہ_ناپ_log2;
    uintptr_t      جڑ           = p_atomic_read(&arr->جڑ);
    إذا (unlikely(!جڑ)) {
        بنانشانی جڑ_level = 0;
        uint64_t idx_iter   = idx >> گرہ_ناپ_log2;
        طالما (idx_iter) {
            idx_iter >>= گرہ_ناپ_log2;
            جڑ_level++;
        }
        uintptr_t new_جڑ = util_sparse_array_گرہ_alloc(arr, جڑ_level);
        جڑ               = util_sparse_array_سیٹ_or_free_گرہ(&arr->جڑ, NULL_گرہ, new_جڑ);
    }

    طالما (1) {
        بنانشانی جڑ_level = util_sparse_array_گرہ_level(جڑ);
        uint64_t جڑ_idx   = idx >> (جڑ_level * گرہ_ناپ_log2);
        إذا (likely(جڑ_idx < (1ull << گرہ_ناپ_log2))) {
            اكسر;
        }

        /* In this case, we have a root but its level is low enough that the
       * requested index is out-of-bounds.
       */
        uintptr_t new_جڑ = util_sparse_array_گرہ_alloc(arr, جڑ_level + 1);

        uintptr_t * new_جڑ_children = (uintptr_t *) util_sparse_array_گرہ_انکڑے(new_جڑ);
        new_جڑ_children[0]          = جڑ;

        /* We only add one at a time instead of the whole tree because it's
       * easier to ensure correctness of both the tree building and the
       * clean-up path.  Because we're only adding one node we never have to
       * worry about trying to free multiple things without freeing the old
       * things.
       */
        جڑ = util_sparse_array_سیٹ_or_free_گرہ(&arr->جڑ, جڑ, new_جڑ);
    }

    فراغ *   گرہ_انکڑے  = util_sparse_array_گرہ_انکڑے(جڑ);
    بنانشانی گرہ_level = util_sparse_array_گرہ_level(جڑ);
    طالما (گرہ_level > 0) {
        uint64_t بچہ_idx = (idx >> (گرہ_level * گرہ_ناپ_log2)) & ((1ull << گرہ_ناپ_log2) - 1);

        uintptr_t * children = (uintptr_t *) گرہ_انکڑے;
        uintptr_t   بچہ    = p_atomic_read(&children[بچہ_idx]);

        إذا (unlikely(!بچہ)) {
            بچہ = util_sparse_array_گرہ_alloc(arr, گرہ_level - 1);
            بچہ = util_sparse_array_سیٹ_or_free_گرہ(&children[بچہ_idx], NULL_گرہ, بچہ);
        }

        گرہ_انکڑے  = util_sparse_array_گرہ_انکڑے(بچہ);
        گرہ_level = util_sparse_array_گرہ_level(بچہ);
    }

    uint64_t elem_idx = idx & ((1ull << گرہ_ناپ_log2) - 1);
    أرجع (فراغ *) ((حرف *) گرہ_انکڑے + (elem_idx * arr->elem_ناپ));
}
