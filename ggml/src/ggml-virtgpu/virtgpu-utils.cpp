#ضمن "virtgpu-utils.h"

#ضمن <malloc.h>
#ضمن <stdlib.h>

#ضمن <cstring>

#تعریف_کن گره_ALLOC_ALIGN 64
#تعریف_کن گره_PTR_ماسک    (~((uintptr_t) گره_ALLOC_ALIGN - 1))
#تعریف_کن گره_LEVEL_ماسک  ((uintptr_t) گره_ALLOC_ALIGN - 1)
#تعریف_کن NULL_گره        0

#تعریف_کن os_malloc_aligned(اندازه_2, _align) _aligned_malloc(اندازه_2, _align)
#تعریف_کن os_free_aligned(_ptr)            حرر(_ptr)
#تعریف_کن p_atomic_cmpxchg(v, old, _new)   __sync_val_compare_and_swap((v), (old), (_new))

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
#پایان_اگر
}

فراغ util_sparse_array_init(util_sparse_array * arr, size_t elem_اندازه, size_t گره_اندازه) {
    memset(arr, 0, حجم(*arr));
    arr->elem_اندازه      = elem_اندازه;
    arr->گره_اندازه_log2 = util_logbase2_64(گره_اندازه);
    أكد(گره_اندازه >= 2 && گره_اندازه == (1ull << arr->گره_اندازه_log2));
}

ساكن مضمن فراغ * os_malloc_aligned(size_t size, size_t alignment) {
    فراغ * ptr;
    alignment = (alignment + حجم(فراغ *) - 1) & ~(حجم(فراغ *) - 1);
    إذا (posix_memalign(&ptr, alignment, size) != 0) {
        أرجع عدم;
    }
    أرجع ptr;
}

ساكن مضمن فراغ * util_sparse_array_گره_داده(uintptr_t handle) {
    أرجع (فراغ *) (handle & گره_PTR_ماسک);
}

ساكن مضمن بیعلامت util_sparse_array_گره_level(uintptr_t handle) {
    أرجع handle & گره_LEVEL_ماسک;
}

ساكن مضمن فراغ util_sparse_array_گره_پایان(util_sparse_array * arr, uintptr_t گره) {
    إذا (util_sparse_array_گره_level(گره) > 0) {
        uintptr_t * children  = (uintptr_t *) util_sparse_array_گره_داده(گره);
        size_t      گره_اندازه = 1ull << arr->گره_اندازه_log2;
        لكل (size_t i = 0; i < گره_اندازه; i++) {
            إذا (children[i]) {
                util_sparse_array_گره_پایان(arr, children[i]);
            }
        }
    }

    os_free_aligned(util_sparse_array_گره_داده(گره));
}

ساكن مضمن uintptr_t util_sparse_array_گره(فراغ * داده, بیعلامت level) {
    أكد(داده != عدم);
    أكد(((uintptr_t) داده & گره_LEVEL_ماسک) == 0);
    أكد((level & گره_PTR_ماسک) == 0);
    أرجع (uintptr_t) داده | level;
}

مضمن uintptr_t util_sparse_array_گره_alloc(util_sparse_array * arr, بیعلامت level) {
    size_t size;
    إذا (level == 0) {
        size = arr->elem_اندازه << arr->گره_اندازه_log2;
    } وإلا {
        size = حجم(uintptr_t) << arr->گره_اندازه_log2;
    }

    فراغ * داده = os_malloc_aligned(size, گره_ALLOC_ALIGN);
    memset(داده, 0, size);

    أرجع util_sparse_array_گره(داده, level);
}

ساكن مضمن uintptr_t util_sparse_array_مجموعه_or_free_گره(uintptr_t * گره_ptr, uintptr_t cmp_گره, uintptr_t گره) {
    uintptr_t prev_گره = p_atomic_cmpxchg(گره_ptr, cmp_گره, گره);

    إذا (prev_گره != cmp_گره) {
        /* We lost the race.  Free this one and return the one that was already
       * allocated.
       */
        os_free_aligned(util_sparse_array_گره_داده(گره));
        أرجع prev_گره;
    } وإلا {
        أرجع گره;
    }
}

فراغ * util_sparse_array_get(util_sparse_array * arr, uint64_t idx) {
    ثابت بیعلامت گره_اندازه_log2 = arr->گره_اندازه_log2;
    uintptr_t      ریشه           = p_atomic_read(&arr->ریشه);
    إذا (unlikely(!ریشه)) {
        بیعلامت ریشه_level = 0;
        uint64_t idx_iter   = idx >> گره_اندازه_log2;
        طالما (idx_iter) {
            idx_iter >>= گره_اندازه_log2;
            ریشه_level++;
        }
        uintptr_t new_ریشه = util_sparse_array_گره_alloc(arr, ریشه_level);
        ریشه               = util_sparse_array_مجموعه_or_free_گره(&arr->ریشه, NULL_گره, new_ریشه);
    }

    طالما (1) {
        بیعلامت ریشه_level = util_sparse_array_گره_level(ریشه);
        uint64_t ریشه_idx   = idx >> (ریشه_level * گره_اندازه_log2);
        إذا (likely(ریشه_idx < (1ull << گره_اندازه_log2))) {
            اكسر;
        }

        /* In this case, we have a root but its level is low enough that the
       * requested index is out-of-bounds.
       */
        uintptr_t new_ریشه = util_sparse_array_گره_alloc(arr, ریشه_level + 1);

        uintptr_t * new_ریشه_children = (uintptr_t *) util_sparse_array_گره_داده(new_ریشه);
        new_ریشه_children[0]          = ریشه;

        /* We only add one at a time instead of the whole tree because it's
       * easier to ensure correctness of both the tree building and the
       * clean-up path.  Because we're only adding one node we never have to
       * worry about trying to free multiple things without freeing the old
       * things.
       */
        ریشه = util_sparse_array_مجموعه_or_free_گره(&arr->ریشه, ریشه, new_ریشه);
    }

    فراغ *   گره_داده  = util_sparse_array_گره_داده(ریشه);
    بیعلامت گره_level = util_sparse_array_گره_level(ریشه);
    طالما (گره_level > 0) {
        uint64_t فرزند_idx = (idx >> (گره_level * گره_اندازه_log2)) & ((1ull << گره_اندازه_log2) - 1);

        uintptr_t * children = (uintptr_t *) گره_داده;
        uintptr_t   فرزند    = p_atomic_read(&children[فرزند_idx]);

        إذا (unlikely(!فرزند)) {
            فرزند = util_sparse_array_گره_alloc(arr, گره_level - 1);
            فرزند = util_sparse_array_مجموعه_or_free_گره(&children[فرزند_idx], NULL_گره, فرزند);
        }

        گره_داده  = util_sparse_array_گره_داده(فرزند);
        گره_level = util_sparse_array_گره_level(فرزند);
    }

    uint64_t elem_idx = idx & ((1ull << گره_اندازه_log2) - 1);
    أرجع (فراغ *) ((حرف *) گره_داده + (elem_idx * arr->elem_اندازه));
}
