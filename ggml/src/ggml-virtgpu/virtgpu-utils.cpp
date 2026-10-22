#அந்தர்பாவ "virtgpu-utils.h"

#அந்தர்பாவ <malloc.h>
#அந்தர்பாவ <stdlib.h>

#அந்தர்பாவ <cstring>

#define முனை_ALLOC_ALIGN 64
#define முனை_PTR_MASK    (~((uintptr_t) முனை_ALLOC_ALIGN - 1))
#define முனை_LEVEL_MASK  ((uintptr_t) முனை_ALLOC_ALIGN - 1)
#define NULL_முனை        0

#define os_malloc_aligned(_size, _align) _aligned_malloc(_size, _align)
#define os_free_aligned(_ptr)            விடுவி(_ptr)
#define p_atomic_cmpxchg(v, old, _new)   __sync_val_compare_and_swap((v), (old), (_new))

நிலையான உள்வரி uint64_t util_logbase2_64(uint64_t n) {
#எனில் defined(HAVE___BUILTIN_CLZLL)
    திரும்பு ((அளவு(uint64_t) * 8 - 1) - __builtin_clzll(n | 1));
#இல்லையெனில்
    uint64_t pos = 0ull;
    எனில் (n >= 1ull << 32) {
        n >>= 32;
        pos += 32;
    }
    எனில் (n >= 1ull << 16) {
        n >>= 16;
        pos += 16;
    }
    எனில் (n >= 1ull << 8) {
        n >>= 8;
        pos += 8;
    }
    எனில் (n >= 1ull << 4) {
        n >>= 4;
        pos += 4;
    }
    எனில் (n >= 1ull << 2) {
        n >>= 2;
        pos += 2;
    }
    எனில் (n >= 1ull << 1) {
        pos += 1;
    }
    திரும்பு pos;
#endif
}

வெறுமை util_sparse_array_init(util_sparse_array * arr, size_t elem_size, size_t முனை_size) {
    memset(arr, 0, அளவு(*arr));
    arr->elem_size      = elem_size;
    arr->முனை_size_log2 = util_logbase2_64(முனை_size);
    ப்ரதிஜ்ஞா(முனை_size >= 2 && முனை_size == (1ull << arr->முனை_size_log2));
}

நிலையான உள்வரி வெறுமை * os_malloc_aligned(size_t size, size_t alignment) {
    வெறுமை * ptr;
    alignment = (alignment + அளவு(வெறுமை *) - 1) & ~(அளவு(வெறுமை *) - 1);
    எனில் (posix_memalign(&ptr, alignment, size) != 0) {
        திரும்பு வெற்று;
    }
    திரும்பு ptr;
}

நிலையான உள்வரி வெறுமை * util_sparse_array_முனை_data(uintptr_t handle) {
    திரும்பு (வெறுமை *) (handle & முனை_PTR_MASK);
}

நிலையான உள்வரி unsigned util_sparse_array_முனை_level(uintptr_t handle) {
    திரும்பு handle & முனை_LEVEL_MASK;
}

நிலையான உள்வரி வெறுமை util_sparse_array_முனை_finish(util_sparse_array * arr, uintptr_t முனை) {
    எனில் (util_sparse_array_முனை_level(முனை) > 0) {
        uintptr_t * children  = (uintptr_t *) util_sparse_array_முனை_data(முனை);
        size_t      முனை_size = 1ull << arr->முனை_size_log2;
        க்கு (size_t i = 0; i < முனை_size; i++) {
            எனில் (children[i]) {
                util_sparse_array_முனை_finish(arr, children[i]);
            }
        }
    }

    os_free_aligned(util_sparse_array_முனை_data(முனை));
}

நிலையான உள்வரி uintptr_t util_sparse_array_முனை(வெறுமை * data, unsigned level) {
    ப்ரதிஜ்ஞா(data != வெற்று);
    ப்ரதிஜ்ஞா(((uintptr_t) data & முனை_LEVEL_MASK) == 0);
    ப்ரதிஜ்ஞா((level & முனை_PTR_MASK) == 0);
    திரும்பு (uintptr_t) data | level;
}

உள்வரி uintptr_t util_sparse_array_முனை_alloc(util_sparse_array * arr, unsigned level) {
    size_t size;
    எனில் (level == 0) {
        size = arr->elem_size << arr->முனை_size_log2;
    } இல்லையெனில் {
        size = அளவு(uintptr_t) << arr->முனை_size_log2;
    }

    வெறுமை * data = os_malloc_aligned(size, முனை_ALLOC_ALIGN);
    memset(data, 0, size);

    திரும்பு util_sparse_array_முனை(data, level);
}

நிலையான உள்வரி uintptr_t util_sparse_array_set_or_free_முனை(uintptr_t * முனை_ptr, uintptr_t cmp_முனை, uintptr_t முனை) {
    uintptr_t prev_முனை = p_atomic_cmpxchg(முனை_ptr, cmp_முனை, முனை);

    எனில் (prev_முனை != cmp_முனை) {
        /* We lost the race.  Free this one and return the one that was already
       * allocated.
       */
        os_free_aligned(util_sparse_array_முனை_data(முனை));
        திரும்பு prev_முனை;
    } இல்லையெனில் {
        திரும்பு முனை;
    }
}

வெறுமை * util_sparse_array_get(util_sparse_array * arr, uint64_t idx) {
    மாறிலி unsigned முனை_size_log2 = arr->முனை_size_log2;
    uintptr_t      வேர்           = p_atomic_read(&arr->வேர்);
    எனில் (unlikely(!வேர்)) {
        unsigned வேர்_level = 0;
        uint64_t idx_iter   = idx >> முனை_size_log2;
        போது (idx_iter) {
            idx_iter >>= முனை_size_log2;
            வேர்_level++;
        }
        uintptr_t new_வேர் = util_sparse_array_முனை_alloc(arr, வேர்_level);
        வேர்               = util_sparse_array_set_or_free_முனை(&arr->வேர், NULL_முனை, new_வேர்);
    }

    போது (1) {
        unsigned வேர்_level = util_sparse_array_முனை_level(வேர்);
        uint64_t வேர்_idx   = idx >> (வேர்_level * முனை_size_log2);
        எனில் (likely(வேர்_idx < (1ull << முனை_size_log2))) {
            நிறுத்து;
        }

        /* In this case, we have a root but its level is low enough that the
       * requested index is out-of-bounds.
       */
        uintptr_t new_வேர் = util_sparse_array_முனை_alloc(arr, வேர்_level + 1);

        uintptr_t * new_வேர்_children = (uintptr_t *) util_sparse_array_முனை_data(new_வேர்);
        new_வேர்_children[0]          = வேர்;

        /* We only add one at a time instead of the whole tree because it's
       * easier to ensure correctness of both the tree building and the
       * clean-up path.  Because we're only adding one node we never have to
       * worry about trying to free multiple things without freeing the old
       * things.
       */
        வேர் = util_sparse_array_set_or_free_முனை(&arr->வேர், வேர், new_வேர்);
    }

    வெறுமை *   முனை_data  = util_sparse_array_முனை_data(வேர்);
    unsigned முனை_level = util_sparse_array_முனை_level(வேர்);
    போது (முனை_level > 0) {
        uint64_t child_idx = (idx >> (முனை_level * முனை_size_log2)) & ((1ull << முனை_size_log2) - 1);

        uintptr_t * children = (uintptr_t *) முனை_data;
        uintptr_t   child    = p_atomic_read(&children[child_idx]);

        எனில் (unlikely(!child)) {
            child = util_sparse_array_முனை_alloc(arr, முனை_level - 1);
            child = util_sparse_array_set_or_free_முனை(&children[child_idx], NULL_முனை, child);
        }

        முனை_data  = util_sparse_array_முனை_data(child);
        முனை_level = util_sparse_array_முனை_level(child);
    }

    uint64_t elem_idx = idx & ((1ull << முனை_size_log2) - 1);
    திரும்பு (வெறுமை *) ((எழுத்து *) முனை_data + (elem_idx * arr->elem_size));
}
