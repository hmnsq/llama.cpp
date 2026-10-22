#ਅਨ੍ਤਰ੍ਭਾਵ "virtgpu-utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <malloc.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਗੰਢ_ALLOC_ALIGN 64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਗੰਢ_PTR_ਨਕਾਬ    (~((uintptr_t) ਗੰਢ_ALLOC_ALIGN - 1))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਗੰਢ_LEVEL_ਨਕਾਬ  ((uintptr_t) ਗੰਢ_ALLOC_ALIGN - 1)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NULL_ਗੰਢ        0

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ os_malloc_aligned(ਮਾਪ_2, _align) _aligned_malloc(ਮਾਪ_2, _align)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ os_free_aligned(_ptr)            ਮੁਕਤ(_ptr)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ p_atomic_cmpxchg(v, old, _new)   __sync_val_compare_and_swap((v), (old), (_new))

ਸਥਿਰ ਸਤਰਅੰਦਰ uint64_t util_logbase2_64(uint64_t n) {
#ਜੇ defined(HAVE___BUILTIN_CLZLL)
    ਵਾਪਸ ((ਆਕਾਰ(uint64_t) * 8 - 1) - __builtin_clzll(n | 1));
#ਨਹੀਂਤਾਂ
    uint64_t pos = 0ull;
    ਜੇ (n >= 1ull << 32) {
        n >>= 32;
        pos += 32;
    }
    ਜੇ (n >= 1ull << 16) {
        n >>= 16;
        pos += 16;
    }
    ਜੇ (n >= 1ull << 8) {
        n >>= 8;
        pos += 8;
    }
    ਜੇ (n >= 1ull << 4) {
        n >>= 4;
        pos += 4;
    }
    ਜੇ (n >= 1ull << 2) {
        n >>= 2;
        pos += 2;
    }
    ਜੇ (n >= 1ull << 1) {
        pos += 1;
    }
    ਵਾਪਸ pos;
#ਜੇ_ਅੰਤ
}

ਖਾਲੀ util_sparse_array_init(util_sparse_array * arr, size_t elem_ਮਾਪ, size_t ਗੰਢ_ਮਾਪ) {
    memset(arr, 0, ਆਕਾਰ(*arr));
    arr->elem_ਮਾਪ      = elem_ਮਾਪ;
    arr->ਗੰਢ_ਮਾਪ_log2 = util_logbase2_64(ਗੰਢ_ਮਾਪ);
    ਪ੍ਰਤਿਜ੍ਞਾ(ਗੰਢ_ਮਾਪ >= 2 && ਗੰਢ_ਮਾਪ == (1ull << arr->ਗੰਢ_ਮਾਪ_log2));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ * os_malloc_aligned(size_t size, size_t alignment) {
    ਖਾਲੀ * ptr;
    alignment = (alignment + ਆਕਾਰ(ਖਾਲੀ *) - 1) & ~(ਆਕਾਰ(ਖਾਲੀ *) - 1);
    ਜੇ (posix_memalign(&ptr, alignment, size) != 0) {
        ਵਾਪਸ ਨਿਰਾ;
    }
    ਵਾਪਸ ptr;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ * util_sparse_array_ਗੰਢ_ਅੰਕੜੇ(uintptr_t handle) {
    ਵਾਪਸ (ਖਾਲੀ *) (handle & ਗੰਢ_PTR_ਨਕਾਬ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਬਿਨਾਨਿਸ਼ਾਨੀ util_sparse_array_ਗੰਢ_level(uintptr_t handle) {
    ਵਾਪਸ handle & ਗੰਢ_LEVEL_ਨਕਾਬ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ util_sparse_array_ਗੰਢ_ਅੰਤ(util_sparse_array * arr, uintptr_t ਗੰਢ) {
    ਜੇ (util_sparse_array_ਗੰਢ_level(ਗੰਢ) > 0) {
        uintptr_t * children  = (uintptr_t *) util_sparse_array_ਗੰਢ_ਅੰਕੜੇ(ਗੰਢ);
        size_t      ਗੰਢ_ਮਾਪ = 1ull << arr->ਗੰਢ_ਮਾਪ_log2;
        ਲਈ (size_t i = 0; i < ਗੰਢ_ਮਾਪ; i++) {
            ਜੇ (children[i]) {
                util_sparse_array_ਗੰਢ_ਅੰਤ(arr, children[i]);
            }
        }
    }

    os_free_aligned(util_sparse_array_ਗੰਢ_ਅੰਕੜੇ(ਗੰਢ));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uintptr_t util_sparse_array_ਗੰਢ(ਖਾਲੀ * ਅੰਕੜੇ, ਬਿਨਾਨਿਸ਼ਾਨੀ level) {
    ਪ੍ਰਤਿਜ੍ਞਾ(ਅੰਕੜੇ != ਨਿਰਾ);
    ਪ੍ਰਤਿਜ੍ਞਾ(((uintptr_t) ਅੰਕੜੇ & ਗੰਢ_LEVEL_ਨਕਾਬ) == 0);
    ਪ੍ਰਤਿਜ੍ਞਾ((level & ਗੰਢ_PTR_ਨਕਾਬ) == 0);
    ਵਾਪਸ (uintptr_t) ਅੰਕੜੇ | level;
}

ਸਤਰਅੰਦਰ uintptr_t util_sparse_array_ਗੰਢ_alloc(util_sparse_array * arr, ਬਿਨਾਨਿਸ਼ਾਨੀ level) {
    size_t size;
    ਜੇ (level == 0) {
        size = arr->elem_ਮਾਪ << arr->ਗੰਢ_ਮਾਪ_log2;
    } ਨਹੀਂਤਾਂ {
        size = ਆਕਾਰ(uintptr_t) << arr->ਗੰਢ_ਮਾਪ_log2;
    }

    ਖਾਲੀ * ਅੰਕੜੇ = os_malloc_aligned(size, ਗੰਢ_ALLOC_ALIGN);
    memset(ਅੰਕੜੇ, 0, size);

    ਵਾਪਸ util_sparse_array_ਗੰਢ(ਅੰਕੜੇ, level);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uintptr_t util_sparse_array_ਸੈੱਟ_or_free_ਗੰਢ(uintptr_t * ਗੰਢ_ptr, uintptr_t cmp_ਗੰਢ, uintptr_t ਗੰਢ) {
    uintptr_t prev_ਗੰਢ = p_atomic_cmpxchg(ਗੰਢ_ptr, cmp_ਗੰਢ, ਗੰਢ);

    ਜੇ (prev_ਗੰਢ != cmp_ਗੰਢ) {
        /* We lost the race.  Free this one and return the one that was already
       * allocated.
       */
        os_free_aligned(util_sparse_array_ਗੰਢ_ਅੰਕੜੇ(ਗੰਢ));
        ਵਾਪਸ prev_ਗੰਢ;
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ ਗੰਢ;
    }
}

ਖਾਲੀ * util_sparse_array_get(util_sparse_array * arr, uint64_t idx) {
    ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਗੰਢ_ਮਾਪ_log2 = arr->ਗੰਢ_ਮਾਪ_log2;
    uintptr_t      ਜੜ੍ਹ           = p_atomic_read(&arr->ਜੜ੍ਹ);
    ਜੇ (unlikely(!ਜੜ੍ਹ)) {
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਜੜ੍ਹ_level = 0;
        uint64_t idx_iter   = idx >> ਗੰਢ_ਮਾਪ_log2;
        ਜਦੋਂ (idx_iter) {
            idx_iter >>= ਗੰਢ_ਮਾਪ_log2;
            ਜੜ੍ਹ_level++;
        }
        uintptr_t new_ਜੜ੍ਹ = util_sparse_array_ਗੰਢ_alloc(arr, ਜੜ੍ਹ_level);
        ਜੜ੍ਹ               = util_sparse_array_ਸੈੱਟ_or_free_ਗੰਢ(&arr->ਜੜ੍ਹ, NULL_ਗੰਢ, new_ਜੜ੍ਹ);
    }

    ਜਦੋਂ (1) {
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਜੜ੍ਹ_level = util_sparse_array_ਗੰਢ_level(ਜੜ੍ਹ);
        uint64_t ਜੜ੍ਹ_idx   = idx >> (ਜੜ੍ਹ_level * ਗੰਢ_ਮਾਪ_log2);
        ਜੇ (likely(ਜੜ੍ਹ_idx < (1ull << ਗੰਢ_ਮਾਪ_log2))) {
            ਰੋਕੋ;
        }

        /* In this case, we have a root but its level is low enough that the
       * requested index is out-of-bounds.
       */
        uintptr_t new_ਜੜ੍ਹ = util_sparse_array_ਗੰਢ_alloc(arr, ਜੜ੍ਹ_level + 1);

        uintptr_t * new_ਜੜ੍ਹ_children = (uintptr_t *) util_sparse_array_ਗੰਢ_ਅੰਕੜੇ(new_ਜੜ੍ਹ);
        new_ਜੜ੍ਹ_children[0]          = ਜੜ੍ਹ;

        /* We only add one at a time instead of the whole tree because it's
       * easier to ensure correctness of both the tree building and the
       * clean-up path.  Because we're only adding one node we never have to
       * worry about trying to free multiple things without freeing the old
       * things.
       */
        ਜੜ੍ਹ = util_sparse_array_ਸੈੱਟ_or_free_ਗੰਢ(&arr->ਜੜ੍ਹ, ਜੜ੍ਹ, new_ਜੜ੍ਹ);
    }

    ਖਾਲੀ *   ਗੰਢ_ਅੰਕੜੇ  = util_sparse_array_ਗੰਢ_ਅੰਕੜੇ(ਜੜ੍ਹ);
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਗੰਢ_level = util_sparse_array_ਗੰਢ_level(ਜੜ੍ਹ);
    ਜਦੋਂ (ਗੰਢ_level > 0) {
        uint64_t ਬੱਚਾ_idx = (idx >> (ਗੰਢ_level * ਗੰਢ_ਮਾਪ_log2)) & ((1ull << ਗੰਢ_ਮਾਪ_log2) - 1);

        uintptr_t * children = (uintptr_t *) ਗੰਢ_ਅੰਕੜੇ;
        uintptr_t   ਬੱਚਾ    = p_atomic_read(&children[ਬੱਚਾ_idx]);

        ਜੇ (unlikely(!ਬੱਚਾ)) {
            ਬੱਚਾ = util_sparse_array_ਗੰਢ_alloc(arr, ਗੰਢ_level - 1);
            ਬੱਚਾ = util_sparse_array_ਸੈੱਟ_or_free_ਗੰਢ(&children[ਬੱਚਾ_idx], NULL_ਗੰਢ, ਬੱਚਾ);
        }

        ਗੰਢ_ਅੰਕੜੇ  = util_sparse_array_ਗੰਢ_ਅੰਕੜੇ(ਬੱਚਾ);
        ਗੰਢ_level = util_sparse_array_ਗੰਢ_level(ਬੱਚਾ);
    }

    uint64_t elem_idx = idx & ((1ull << ਗੰਢ_ਮਾਪ_log2) - 1);
    ਵਾਪਸ (ਖਾਲੀ *) ((ਅੱਖਰ *) ਗੰਢ_ਅੰਕੜੇ + (elem_idx * arr->elem_ਮਾਪ));
}
