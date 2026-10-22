#ଅନ୍ତର୍ଭାଵ "virtgpu-utils.h"

#ଅନ୍ତର୍ଭାଵ <malloc.h>
#ଅନ୍ତର୍ଭାଵ <stdlib.h>

#ଅନ୍ତର୍ଭାଵ <cstring>

#define ଗଣ୍ଠି_ALLOC_ALIGN 64
#define ଗଣ୍ଠି_PTR_MASK    (~((uintptr_t) ଗଣ୍ଠି_ALLOC_ALIGN - 1))
#define ଗଣ୍ଠି_LEVEL_MASK  ((uintptr_t) ଗଣ୍ଠି_ALLOC_ALIGN - 1)
#define NULL_ଗଣ୍ଠି        0

#define os_malloc_aligned(_size, _align) _aligned_malloc(_size, _align)
#define os_free_aligned(_ptr)            ମୁକ୍ତ(_ptr)
#define p_atomic_cmpxchg(v, old, _new)   __sync_val_compare_and_swap((v), (old), (_new))

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint64_t util_logbase2_64(uint64_t n) {
#ଯଦି defined(HAVE___BUILTIN_CLZLL)
    ଫେରନ୍ତୁ ((ଆକାର(uint64_t) * 8 - 1) - __builtin_clzll(n | 1));
#ଅନ୍ୟଥା
    uint64_t pos = 0ull;
    ଯଦି (n >= 1ull << 32) {
        n >>= 32;
        pos += 32;
    }
    ଯଦି (n >= 1ull << 16) {
        n >>= 16;
        pos += 16;
    }
    ଯଦି (n >= 1ull << 8) {
        n >>= 8;
        pos += 8;
    }
    ଯଦି (n >= 1ull << 4) {
        n >>= 4;
        pos += 4;
    }
    ଯଦି (n >= 1ull << 2) {
        n >>= 2;
        pos += 2;
    }
    ଯଦି (n >= 1ull << 1) {
        pos += 1;
    }
    ଫେରନ୍ତୁ pos;
#endif
}

ଶୂନ୍ୟ util_sparse_array_init(util_sparse_array * arr, size_t elem_size, size_t ଗଣ୍ଠି_size) {
    memset(arr, 0, ଆକାର(*arr));
    arr->elem_size      = elem_size;
    arr->ଗଣ୍ଠି_size_log2 = util_logbase2_64(ଗଣ୍ଠି_size);
    ପ୍ରତିଜ୍ଞା(ଗଣ୍ଠି_size >= 2 && ଗଣ୍ଠି_size == (1ull << arr->ଗଣ୍ଠି_size_log2));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ * os_malloc_aligned(size_t size, size_t alignment) {
    ଶୂନ୍ୟ * ptr;
    alignment = (alignment + ଆକାର(ଶୂନ୍ୟ *) - 1) & ~(ଆକାର(ଶୂନ୍ୟ *) - 1);
    ଯଦି (posix_memalign(&ptr, alignment, size) != 0) {
        ଫେରନ୍ତୁ ଖାଲି;
    }
    ଫେରନ୍ତୁ ptr;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ * util_sparse_array_ଗଣ୍ଠି_data(uintptr_t handle) {
    ଫେରନ୍ତୁ (ଶୂନ୍ୟ *) (handle & ଗଣ୍ଠି_PTR_MASK);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ unsigned util_sparse_array_ଗଣ୍ଠି_level(uintptr_t handle) {
    ଫେରନ୍ତୁ handle & ଗଣ୍ଠି_LEVEL_MASK;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ util_sparse_array_ଗଣ୍ଠି_finish(util_sparse_array * arr, uintptr_t ଗଣ୍ଠି) {
    ଯଦି (util_sparse_array_ଗଣ୍ଠି_level(ଗଣ୍ଠି) > 0) {
        uintptr_t * children  = (uintptr_t *) util_sparse_array_ଗଣ୍ଠି_data(ଗଣ୍ଠି);
        size_t      ଗଣ୍ଠି_size = 1ull << arr->ଗଣ୍ଠି_size_log2;
        ପାଇଁ (size_t i = 0; i < ଗଣ୍ଠି_size; i++) {
            ଯଦି (children[i]) {
                util_sparse_array_ଗଣ୍ଠି_finish(arr, children[i]);
            }
        }
    }

    os_free_aligned(util_sparse_array_ଗଣ୍ଠି_data(ଗଣ୍ଠି));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uintptr_t util_sparse_array_ଗଣ୍ଠି(ଶୂନ୍ୟ * data, unsigned level) {
    ପ୍ରତିଜ୍ଞା(data != ଖାଲି);
    ପ୍ରତିଜ୍ଞା(((uintptr_t) data & ଗଣ୍ଠି_LEVEL_MASK) == 0);
    ପ୍ରତିଜ୍ଞା((level & ଗଣ୍ଠି_PTR_MASK) == 0);
    ଫେରନ୍ତୁ (uintptr_t) data | level;
}

ଧାଡ଼ିଭିତରେ uintptr_t util_sparse_array_ଗଣ୍ଠି_alloc(util_sparse_array * arr, unsigned level) {
    size_t size;
    ଯଦି (level == 0) {
        size = arr->elem_size << arr->ଗଣ୍ଠି_size_log2;
    } ଅନ୍ୟଥା {
        size = ଆକାର(uintptr_t) << arr->ଗଣ୍ଠି_size_log2;
    }

    ଶୂନ୍ୟ * data = os_malloc_aligned(size, ଗଣ୍ଠି_ALLOC_ALIGN);
    memset(data, 0, size);

    ଫେରନ୍ତୁ util_sparse_array_ଗଣ୍ଠି(data, level);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uintptr_t util_sparse_array_set_or_free_ଗଣ୍ଠି(uintptr_t * ଗଣ୍ଠି_ptr, uintptr_t cmp_ଗଣ୍ଠି, uintptr_t ଗଣ୍ଠି) {
    uintptr_t prev_ଗଣ୍ଠି = p_atomic_cmpxchg(ଗଣ୍ଠି_ptr, cmp_ଗଣ୍ଠି, ଗଣ୍ଠି);

    ଯଦି (prev_ଗଣ୍ଠି != cmp_ଗଣ୍ଠି) {
        /* We lost the race.  Free this one and return the one that was already
       * allocated.
       */
        os_free_aligned(util_sparse_array_ଗଣ୍ଠି_data(ଗଣ୍ଠି));
        ଫେରନ୍ତୁ prev_ଗଣ୍ଠି;
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ ଗଣ୍ଠି;
    }
}

ଶୂନ୍ୟ * util_sparse_array_get(util_sparse_array * arr, uint64_t idx) {
    ଅଚଳ unsigned ଗଣ୍ଠି_size_log2 = arr->ଗଣ୍ଠି_size_log2;
    uintptr_t      ମୂଳ           = p_atomic_read(&arr->ମୂଳ);
    ଯଦି (unlikely(!ମୂଳ)) {
        unsigned ମୂଳ_level = 0;
        uint64_t idx_iter   = idx >> ଗଣ୍ଠି_size_log2;
        ଯେତେବେଳେ (idx_iter) {
            idx_iter >>= ଗଣ୍ଠି_size_log2;
            ମୂଳ_level++;
        }
        uintptr_t new_ମୂଳ = util_sparse_array_ଗଣ୍ଠି_alloc(arr, ମୂଳ_level);
        ମୂଳ               = util_sparse_array_set_or_free_ଗଣ୍ଠି(&arr->ମୂଳ, NULL_ଗଣ୍ଠି, new_ମୂଳ);
    }

    ଯେତେବେଳେ (1) {
        unsigned ମୂଳ_level = util_sparse_array_ଗଣ୍ଠି_level(ମୂଳ);
        uint64_t ମୂଳ_idx   = idx >> (ମୂଳ_level * ଗଣ୍ଠି_size_log2);
        ଯଦି (likely(ମୂଳ_idx < (1ull << ଗଣ୍ଠି_size_log2))) {
            ଭାଙ୍ଗ;
        }

        /* In this case, we have a root but its level is low enough that the
       * requested index is out-of-bounds.
       */
        uintptr_t new_ମୂଳ = util_sparse_array_ଗଣ୍ଠି_alloc(arr, ମୂଳ_level + 1);

        uintptr_t * new_ମୂଳ_children = (uintptr_t *) util_sparse_array_ଗଣ୍ଠି_data(new_ମୂଳ);
        new_ମୂଳ_children[0]          = ମୂଳ;

        /* We only add one at a time instead of the whole tree because it's
       * easier to ensure correctness of both the tree building and the
       * clean-up path.  Because we're only adding one node we never have to
       * worry about trying to free multiple things without freeing the old
       * things.
       */
        ମୂଳ = util_sparse_array_set_or_free_ଗଣ୍ଠି(&arr->ମୂଳ, ମୂଳ, new_ମୂଳ);
    }

    ଶୂନ୍ୟ *   ଗଣ୍ଠି_data  = util_sparse_array_ଗଣ୍ଠି_data(ମୂଳ);
    unsigned ଗଣ୍ଠି_level = util_sparse_array_ଗଣ୍ଠି_level(ମୂଳ);
    ଯେତେବେଳେ (ଗଣ୍ଠି_level > 0) {
        uint64_t child_idx = (idx >> (ଗଣ୍ଠି_level * ଗଣ୍ଠି_size_log2)) & ((1ull << ଗଣ୍ଠି_size_log2) - 1);

        uintptr_t * children = (uintptr_t *) ଗଣ୍ଠି_data;
        uintptr_t   child    = p_atomic_read(&children[child_idx]);

        ଯଦି (unlikely(!child)) {
            child = util_sparse_array_ଗଣ୍ଠି_alloc(arr, ଗଣ୍ଠି_level - 1);
            child = util_sparse_array_set_or_free_ଗଣ୍ଠି(&children[child_idx], NULL_ଗଣ୍ଠି, child);
        }

        ଗଣ୍ଠି_data  = util_sparse_array_ଗଣ୍ଠି_data(child);
        ଗଣ୍ଠି_level = util_sparse_array_ଗଣ୍ଠି_level(child);
    }

    uint64_t elem_idx = idx & ((1ull << ଗଣ୍ଠି_size_log2) - 1);
    ଫେରନ୍ତୁ (ଶୂନ୍ୟ *) ((ଅକ୍ଷର *) ଗଣ୍ଠି_data + (elem_idx * arr->elem_size));
}
