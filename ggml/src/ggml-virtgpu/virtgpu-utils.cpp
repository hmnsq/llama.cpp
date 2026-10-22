#અન્તર્ભાવ "virtgpu-utils.h"

#અન્તર્ભાવ <malloc.h>
#અન્તર્ભાવ <stdlib.h>

#અન્તર્ભાવ <cstring>

#define ગાંઠ_ALLOC_ALIGN 64
#define ગાંઠ_PTR_MASK    (~((uintptr_t) ગાંઠ_ALLOC_ALIGN - 1))
#define ગાંઠ_LEVEL_MASK  ((uintptr_t) ગાંઠ_ALLOC_ALIGN - 1)
#define NULL_ગાંઠ        0

#define os_malloc_aligned(_size, _align) _aligned_malloc(_size, _align)
#define os_free_aligned(_ptr)            મુક્ત(_ptr)
#define p_atomic_cmpxchg(v, old, _new)   __sync_val_compare_and_swap((v), (old), (_new))

સ્થિર અંતઃરેખ uint64_t util_logbase2_64(uint64_t n) {
#જો defined(HAVE___BUILTIN_CLZLL)
    પાછા ((કદ(uint64_t) * 8 - 1) - __builtin_clzll(n | 1));
#નહીંતર
    uint64_t pos = 0ull;
    જો (n >= 1ull << 32) {
        n >>= 32;
        pos += 32;
    }
    જો (n >= 1ull << 16) {
        n >>= 16;
        pos += 16;
    }
    જો (n >= 1ull << 8) {
        n >>= 8;
        pos += 8;
    }
    જો (n >= 1ull << 4) {
        n >>= 4;
        pos += 4;
    }
    જો (n >= 1ull << 2) {
        n >>= 2;
        pos += 2;
    }
    જો (n >= 1ull << 1) {
        pos += 1;
    }
    પાછા pos;
#endif
}

શૂન્ય util_sparse_array_init(util_sparse_array * arr, size_t elem_size, size_t ગાંઠ_size) {
    memset(arr, 0, કદ(*arr));
    arr->elem_size      = elem_size;
    arr->ગાંઠ_size_log2 = util_logbase2_64(ગાંઠ_size);
    પ્રતિજ્ઞા(ગાંઠ_size >= 2 && ગાંઠ_size == (1ull << arr->ગાંઠ_size_log2));
}

સ્થિર અંતઃરેખ શૂન્ય * os_malloc_aligned(size_t size, size_t alignment) {
    શૂન્ય * ptr;
    alignment = (alignment + કદ(શૂન્ય *) - 1) & ~(કદ(શૂન્ય *) - 1);
    જો (posix_memalign(&ptr, alignment, size) != 0) {
        પાછા ખાલી;
    }
    પાછા ptr;
}

સ્થિર અંતઃરેખ શૂન્ય * util_sparse_array_ગાંઠ_data(uintptr_t handle) {
    પાછા (શૂન્ય *) (handle & ગાંઠ_PTR_MASK);
}

સ્થિર અંતઃરેખ unsigned util_sparse_array_ગાંઠ_level(uintptr_t handle) {
    પાછા handle & ગાંઠ_LEVEL_MASK;
}

સ્થિર અંતઃરેખ શૂન્ય util_sparse_array_ગાંઠ_finish(util_sparse_array * arr, uintptr_t ગાંઠ) {
    જો (util_sparse_array_ગાંઠ_level(ગાંઠ) > 0) {
        uintptr_t * children  = (uintptr_t *) util_sparse_array_ગાંઠ_data(ગાંઠ);
        size_t      ગાંઠ_size = 1ull << arr->ગાંઠ_size_log2;
        માટે (size_t i = 0; i < ગાંઠ_size; i++) {
            જો (children[i]) {
                util_sparse_array_ગાંઠ_finish(arr, children[i]);
            }
        }
    }

    os_free_aligned(util_sparse_array_ગાંઠ_data(ગાંઠ));
}

સ્થિર અંતઃરેખ uintptr_t util_sparse_array_ગાંઠ(શૂન્ય * data, unsigned level) {
    પ્રતિજ્ઞા(data != ખાલી);
    પ્રતિજ્ઞા(((uintptr_t) data & ગાંઠ_LEVEL_MASK) == 0);
    પ્રતિજ્ઞા((level & ગાંઠ_PTR_MASK) == 0);
    પાછા (uintptr_t) data | level;
}

અંતઃરેખ uintptr_t util_sparse_array_ગાંઠ_alloc(util_sparse_array * arr, unsigned level) {
    size_t size;
    જો (level == 0) {
        size = arr->elem_size << arr->ગાંઠ_size_log2;
    } નહીંતર {
        size = કદ(uintptr_t) << arr->ગાંઠ_size_log2;
    }

    શૂન્ય * data = os_malloc_aligned(size, ગાંઠ_ALLOC_ALIGN);
    memset(data, 0, size);

    પાછા util_sparse_array_ગાંઠ(data, level);
}

સ્થિર અંતઃરેખ uintptr_t util_sparse_array_set_or_free_ગાંઠ(uintptr_t * ગાંઠ_ptr, uintptr_t cmp_ગાંઠ, uintptr_t ગાંઠ) {
    uintptr_t prev_ગાંઠ = p_atomic_cmpxchg(ગાંઠ_ptr, cmp_ગાંઠ, ગાંઠ);

    જો (prev_ગાંઠ != cmp_ગાંઠ) {
        /* We lost the race.  Free this one and return the one that was already
       * allocated.
       */
        os_free_aligned(util_sparse_array_ગાંઠ_data(ગાંઠ));
        પાછા prev_ગાંઠ;
    } નહીંતર {
        પાછા ગાંઠ;
    }
}

શૂન્ય * util_sparse_array_get(util_sparse_array * arr, uint64_t idx) {
    અચલ unsigned ગાંઠ_size_log2 = arr->ગાંઠ_size_log2;
    uintptr_t      મૂળ           = p_atomic_read(&arr->મૂળ);
    જો (unlikely(!મૂળ)) {
        unsigned મૂળ_level = 0;
        uint64_t idx_iter   = idx >> ગાંઠ_size_log2;
        જ્યારે (idx_iter) {
            idx_iter >>= ગાંઠ_size_log2;
            મૂળ_level++;
        }
        uintptr_t new_મૂળ = util_sparse_array_ગાંઠ_alloc(arr, મૂળ_level);
        મૂળ               = util_sparse_array_set_or_free_ગાંઠ(&arr->મૂળ, NULL_ગાંઠ, new_મૂળ);
    }

    જ્યારે (1) {
        unsigned મૂળ_level = util_sparse_array_ગાંઠ_level(મૂળ);
        uint64_t મૂળ_idx   = idx >> (મૂળ_level * ગાંઠ_size_log2);
        જો (likely(મૂળ_idx < (1ull << ગાંઠ_size_log2))) {
            તોડો;
        }

        /* In this case, we have a root but its level is low enough that the
       * requested index is out-of-bounds.
       */
        uintptr_t new_મૂળ = util_sparse_array_ગાંઠ_alloc(arr, મૂળ_level + 1);

        uintptr_t * new_મૂળ_children = (uintptr_t *) util_sparse_array_ગાંઠ_data(new_મૂળ);
        new_મૂળ_children[0]          = મૂળ;

        /* We only add one at a time instead of the whole tree because it's
       * easier to ensure correctness of both the tree building and the
       * clean-up path.  Because we're only adding one node we never have to
       * worry about trying to free multiple things without freeing the old
       * things.
       */
        મૂળ = util_sparse_array_set_or_free_ગાંઠ(&arr->મૂળ, મૂળ, new_મૂળ);
    }

    શૂન્ય *   ગાંઠ_data  = util_sparse_array_ગાંઠ_data(મૂળ);
    unsigned ગાંઠ_level = util_sparse_array_ગાંઠ_level(મૂળ);
    જ્યારે (ગાંઠ_level > 0) {
        uint64_t child_idx = (idx >> (ગાંઠ_level * ગાંઠ_size_log2)) & ((1ull << ગાંઠ_size_log2) - 1);

        uintptr_t * children = (uintptr_t *) ગાંઠ_data;
        uintptr_t   child    = p_atomic_read(&children[child_idx]);

        જો (unlikely(!child)) {
            child = util_sparse_array_ગાંઠ_alloc(arr, ગાંઠ_level - 1);
            child = util_sparse_array_set_or_free_ગાંઠ(&children[child_idx], NULL_ગાંઠ, child);
        }

        ગાંઠ_data  = util_sparse_array_ગાંઠ_data(child);
        ગાંઠ_level = util_sparse_array_ગાંઠ_level(child);
    }

    uint64_t elem_idx = idx & ((1ull << ગાંઠ_size_log2) - 1);
    પાછા (શૂન્ય *) ((અક્ષર *) ગાંઠ_data + (elem_idx * arr->elem_size));
}
