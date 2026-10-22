#ಅನ್ತರ್ಭಾವ "virtgpu-utils.h"

#ಅನ್ತರ್ಭಾವ <malloc.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>

#ಅನ್ತರ್ಭಾವ <cstring>

#ವ್ಯಾಖ್ಯಾನಿಸು ಗಂಟು_ALLOC_ALIGN 64
#ವ್ಯಾಖ್ಯಾನಿಸು ಗಂಟು_PTR_ಮುಖವಾಡ    (~((uintptr_t) ಗಂಟು_ALLOC_ALIGN - 1))
#ವ್ಯಾಖ್ಯಾನಿಸು ಗಂಟು_LEVEL_ಮುಖವಾಡ  ((uintptr_t) ಗಂಟು_ALLOC_ALIGN - 1)
#ವ್ಯಾಖ್ಯಾನಿಸು NULL_ಗಂಟು        0

#ವ್ಯಾಖ್ಯಾನಿಸು os_malloc_aligned(ಗಾತ್ರ_3, _align) _aligned_malloc(ಗಾತ್ರ_3, _align)
#ವ್ಯಾಖ್ಯಾನಿಸು os_free_aligned(_ptr)            ಬಿಡುಗಡೆ(_ptr)
#ವ್ಯಾಖ್ಯಾನಿಸು p_atomic_cmpxchg(v, old, _new)   __sync_val_compare_and_swap((v), (old), (_new))

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint64_t util_logbase2_64(uint64_t n) {
#ಆದರೆ defined(HAVE___BUILTIN_CLZLL)
    ಹಿಂತಿರುಗಿ ((ಗಾತ್ರ(uint64_t) * 8 - 1) - __builtin_clzll(n | 1));
#ಇಲ್ಲದಿದ್ದರೆ
    uint64_t pos = 0ull;
    ಆದರೆ (n >= 1ull << 32) {
        n >>= 32;
        pos += 32;
    }
    ಆದರೆ (n >= 1ull << 16) {
        n >>= 16;
        pos += 16;
    }
    ಆದರೆ (n >= 1ull << 8) {
        n >>= 8;
        pos += 8;
    }
    ಆದರೆ (n >= 1ull << 4) {
        n >>= 4;
        pos += 4;
    }
    ಆದರೆ (n >= 1ull << 2) {
        n >>= 2;
        pos += 2;
    }
    ಆದರೆ (n >= 1ull << 1) {
        pos += 1;
    }
    ಹಿಂತಿರುಗಿ pos;
#ಆದರೆ_ಅಂತ್ಯ
}

ಶೂನ್ಯ util_sparse_array_init(util_sparse_array * arr, size_t elem_ಗಾತ್ರ, size_t ಗಂಟು_ಗಾತ್ರ) {
    memset(arr, 0, ಗಾತ್ರ(*arr));
    arr->elem_ಗಾತ್ರ      = elem_ಗಾತ್ರ;
    arr->ಗಂಟು_ಗಾತ್ರ_log2 = util_logbase2_64(ಗಂಟು_ಗಾತ್ರ);
    ಪ್ರತಿಜ್ಞಾ(ಗಂಟು_ಗಾತ್ರ >= 2 && ಗಂಟು_ಗಾತ್ರ == (1ull << arr->ಗಂಟು_ಗಾತ್ರ_log2));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ * os_malloc_aligned(size_t size, size_t alignment) {
    ಶೂನ್ಯ * ptr;
    alignment = (alignment + ಗಾತ್ರ(ಶೂನ್ಯ *) - 1) & ~(ಗಾತ್ರ(ಶೂನ್ಯ *) - 1);
    ಆದರೆ (posix_memalign(&ptr, alignment, size) != 0) {
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }
    ಹಿಂತಿರುಗಿ ptr;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ * util_sparse_array_ಗಂಟು_ದತ್ತಾಂಶ(uintptr_t handle) {
    ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *) (handle & ಗಂಟು_PTR_ಮುಖವಾಡ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಚಿಹ್ನೆರಹಿತ util_sparse_array_ಗಂಟು_level(uintptr_t handle) {
    ಹಿಂತಿರುಗಿ handle & ಗಂಟು_LEVEL_ಮುಖವಾಡ;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ util_sparse_array_ಗಂಟು_ಮುಕ್ತಾಯ(util_sparse_array * arr, uintptr_t ಗಂಟು) {
    ಆದರೆ (util_sparse_array_ಗಂಟು_level(ಗಂಟು) > 0) {
        uintptr_t * children  = (uintptr_t *) util_sparse_array_ಗಂಟು_ದತ್ತಾಂಶ(ಗಂಟು);
        size_t      ಗಂಟು_ಗಾತ್ರ = 1ull << arr->ಗಂಟು_ಗಾತ್ರ_log2;
        ಗಾಗಿ (size_t i = 0; i < ಗಂಟು_ಗಾತ್ರ; i++) {
            ಆದರೆ (children[i]) {
                util_sparse_array_ಗಂಟು_ಮುಕ್ತಾಯ(arr, children[i]);
            }
        }
    }

    os_free_aligned(util_sparse_array_ಗಂಟು_ದತ್ತಾಂಶ(ಗಂಟು));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uintptr_t util_sparse_array_ಗಂಟು(ಶೂನ್ಯ * ದತ್ತಾಂಶ, ಚಿಹ್ನೆರಹಿತ level) {
    ಪ್ರತಿಜ್ಞಾ(ದತ್ತಾಂಶ != ಖಾಲಿ);
    ಪ್ರತಿಜ್ಞಾ(((uintptr_t) ದತ್ತಾಂಶ & ಗಂಟು_LEVEL_ಮುಖವಾಡ) == 0);
    ಪ್ರತಿಜ್ಞಾ((level & ಗಂಟು_PTR_ಮುಖವಾಡ) == 0);
    ಹಿಂತಿರುಗಿ (uintptr_t) ದತ್ತಾಂಶ | level;
}

ಸಾಲೊಳಗೆ uintptr_t util_sparse_array_ಗಂಟು_alloc(util_sparse_array * arr, ಚಿಹ್ನೆರಹಿತ level) {
    size_t size;
    ಆದರೆ (level == 0) {
        size = arr->elem_ಗಾತ್ರ << arr->ಗಂಟು_ಗಾತ್ರ_log2;
    } ಇಲ್ಲದಿದ್ದರೆ {
        size = ಗಾತ್ರ(uintptr_t) << arr->ಗಂಟು_ಗಾತ್ರ_log2;
    }

    ಶೂನ್ಯ * ದತ್ತಾಂಶ = os_malloc_aligned(size, ಗಂಟು_ALLOC_ALIGN);
    memset(ದತ್ತಾಂಶ, 0, size);

    ಹಿಂತಿರುಗಿ util_sparse_array_ಗಂಟು(ದತ್ತಾಂಶ, level);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uintptr_t util_sparse_array_ಗಣ_or_free_ಗಂಟು(uintptr_t * ಗಂಟು_ptr, uintptr_t cmp_ಗಂಟು, uintptr_t ಗಂಟು) {
    uintptr_t prev_ಗಂಟು = p_atomic_cmpxchg(ಗಂಟು_ptr, cmp_ಗಂಟು, ಗಂಟು);

    ಆದರೆ (prev_ಗಂಟು != cmp_ಗಂಟು) {
        /* We lost the race.  Free this one and return the one that was already
       * allocated.
       */
        os_free_aligned(util_sparse_array_ಗಂಟು_ದತ್ತಾಂಶ(ಗಂಟು));
        ಹಿಂತಿರುಗಿ prev_ಗಂಟು;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ ಗಂಟು;
    }
}

ಶೂನ್ಯ * util_sparse_array_get(util_sparse_array * arr, uint64_t idx) {
    ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಗಂಟು_ಗಾತ್ರ_log2 = arr->ಗಂಟು_ಗಾತ್ರ_log2;
    uintptr_t      ಬೇರು           = p_atomic_read(&arr->ಬೇರು);
    ಆದರೆ (unlikely(!ಬೇರು)) {
        ಚಿಹ್ನೆರಹಿತ ಬೇರು_level = 0;
        uint64_t idx_iter   = idx >> ಗಂಟು_ಗಾತ್ರ_log2;
        ಆಗ (idx_iter) {
            idx_iter >>= ಗಂಟು_ಗಾತ್ರ_log2;
            ಬೇರು_level++;
        }
        uintptr_t new_ಬೇರು = util_sparse_array_ಗಂಟು_alloc(arr, ಬೇರು_level);
        ಬೇರು               = util_sparse_array_ಗಣ_or_free_ಗಂಟು(&arr->ಬೇರು, NULL_ಗಂಟು, new_ಬೇರು);
    }

    ಆಗ (1) {
        ಚಿಹ್ನೆರಹಿತ ಬೇರು_level = util_sparse_array_ಗಂಟು_level(ಬೇರು);
        uint64_t ಬೇರು_idx   = idx >> (ಬೇರು_level * ಗಂಟು_ಗಾತ್ರ_log2);
        ಆದರೆ (likely(ಬೇರು_idx < (1ull << ಗಂಟು_ಗಾತ್ರ_log2))) {
            ನಿಲ್ಲಿಸು;
        }

        /* In this case, we have a root but its level is low enough that the
       * requested index is out-of-bounds.
       */
        uintptr_t new_ಬೇರು = util_sparse_array_ಗಂಟು_alloc(arr, ಬೇರು_level + 1);

        uintptr_t * new_ಬೇರು_children = (uintptr_t *) util_sparse_array_ಗಂಟು_ದತ್ತಾಂಶ(new_ಬೇರು);
        new_ಬೇರು_children[0]          = ಬೇರು;

        /* We only add one at a time instead of the whole tree because it's
       * easier to ensure correctness of both the tree building and the
       * clean-up path.  Because we're only adding one node we never have to
       * worry about trying to free multiple things without freeing the old
       * things.
       */
        ಬೇರು = util_sparse_array_ಗಣ_or_free_ಗಂಟು(&arr->ಬೇರು, ಬೇರು, new_ಬೇರು);
    }

    ಶೂನ್ಯ *   ಗಂಟು_ದತ್ತಾಂಶ  = util_sparse_array_ಗಂಟು_ದತ್ತಾಂಶ(ಬೇರು);
    ಚಿಹ್ನೆರಹಿತ ಗಂಟು_level = util_sparse_array_ಗಂಟು_level(ಬೇರು);
    ಆಗ (ಗಂಟು_level > 0) {
        uint64_t ಮಗು_idx = (idx >> (ಗಂಟು_level * ಗಂಟು_ಗಾತ್ರ_log2)) & ((1ull << ಗಂಟು_ಗಾತ್ರ_log2) - 1);

        uintptr_t * children = (uintptr_t *) ಗಂಟು_ದತ್ತಾಂಶ;
        uintptr_t   ಮಗು    = p_atomic_read(&children[ಮಗು_idx]);

        ಆದರೆ (unlikely(!ಮಗು)) {
            ಮಗು = util_sparse_array_ಗಂಟು_alloc(arr, ಗಂಟು_level - 1);
            ಮಗು = util_sparse_array_ಗಣ_or_free_ಗಂಟು(&children[ಮಗು_idx], NULL_ಗಂಟು, ಮಗು);
        }

        ಗಂಟು_ದತ್ತಾಂಶ  = util_sparse_array_ಗಂಟು_ದತ್ತಾಂಶ(ಮಗು);
        ಗಂಟು_level = util_sparse_array_ಗಂಟು_level(ಮಗು);
    }

    uint64_t elem_idx = idx & ((1ull << ಗಂಟು_ಗಾತ್ರ_log2) - 1);
    ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *) ((ಅಕ್ಷರ *) ಗಂಟು_ದತ್ತಾಂಶ + (elem_idx * arr->elem_ಗಾತ್ರ));
}
