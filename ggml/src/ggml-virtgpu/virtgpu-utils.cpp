#అన్తర్భావ "virtgpu-utils.h"

#అన్తర్భావ <malloc.h>
#అన్తర్భావ <stdlib.h>

#అన్తర్భావ <cstring>

#నిర్వచించు కణుపు_ALLOC_ALIGN 64
#నిర్వచించు కణుపు_PTR_ముసుగు    (~((uintptr_t) కణుపు_ALLOC_ALIGN - 1))
#నిర్వచించు కణుపు_LEVEL_ముసుగు  ((uintptr_t) కణుపు_ALLOC_ALIGN - 1)
#నిర్వచించు NULL_కణుపు        0

#నిర్వచించు os_malloc_aligned(కొలత_2, _align) _aligned_malloc(కొలత_2, _align)
#నిర్వచించు os_free_aligned(_ptr)            విడుదల(_ptr)
#నిర్వచించు p_atomic_cmpxchg(v, old, _new)   __sync_val_compare_and_swap((v), (old), (_new))

స్థిర అంతఃవరుస uint64_t util_logbase2_64(uint64_t n) {
#ఒకవేళ defined(HAVE___BUILTIN_CLZLL)
    తిరిగి ((పరిమాణం(uint64_t) * 8 - 1) - __builtin_clzll(n | 1));
#లేకపోతే
    uint64_t pos = 0ull;
    ఒకవేళ (n >= 1ull << 32) {
        n >>= 32;
        pos += 32;
    }
    ఒకవేళ (n >= 1ull << 16) {
        n >>= 16;
        pos += 16;
    }
    ఒకవేళ (n >= 1ull << 8) {
        n >>= 8;
        pos += 8;
    }
    ఒకవేళ (n >= 1ull << 4) {
        n >>= 4;
        pos += 4;
    }
    ఒకవేళ (n >= 1ull << 2) {
        n >>= 2;
        pos += 2;
    }
    ఒకవేళ (n >= 1ull << 1) {
        pos += 1;
    }
    తిరిగి pos;
#అయితే_ముగింపు
}

శూన్య util_sparse_array_init(util_sparse_array * arr, size_t elem_కొలత, size_t కణుపు_కొలత) {
    memset(arr, 0, పరిమాణం(*arr));
    arr->elem_కొలత      = elem_కొలత;
    arr->కణుపు_కొలత_log2 = util_logbase2_64(కణుపు_కొలత);
    ప్రతిజ్ఞా(కణుపు_కొలత >= 2 && కణుపు_కొలత == (1ull << arr->కణుపు_కొలత_log2));
}

స్థిర అంతఃవరుస శూన్య * os_malloc_aligned(size_t size, size_t alignment) {
    శూన్య * ptr;
    alignment = (alignment + పరిమాణం(శూన్య *) - 1) & ~(పరిమాణం(శూన్య *) - 1);
    ఒకవేళ (posix_memalign(&ptr, alignment, size) != 0) {
        తిరిగి ఖాళీ;
    }
    తిరిగి ptr;
}

స్థిర అంతఃవరుస శూన్య * util_sparse_array_కణుపు_దత్తాంశం(uintptr_t handle) {
    తిరిగి (శూన్య *) (handle & కణుపు_PTR_ముసుగు);
}

స్థిర అంతఃవరుస గుర్తులేని util_sparse_array_కణుపు_level(uintptr_t handle) {
    తిరిగి handle & కణుపు_LEVEL_ముసుగు;
}

స్థిర అంతఃవరుస శూన్య util_sparse_array_కణుపు_ముగింపు(util_sparse_array * arr, uintptr_t కణుపు) {
    ఒకవేళ (util_sparse_array_కణుపు_level(కణుపు) > 0) {
        uintptr_t * children  = (uintptr_t *) util_sparse_array_కణుపు_దత్తాంశం(కణుపు);
        size_t      కణుపు_కొలత = 1ull << arr->కణుపు_కొలత_log2;
        కొరకు (size_t i = 0; i < కణుపు_కొలత; i++) {
            ఒకవేళ (children[i]) {
                util_sparse_array_కణుపు_ముగింపు(arr, children[i]);
            }
        }
    }

    os_free_aligned(util_sparse_array_కణుపు_దత్తాంశం(కణుపు));
}

స్థిర అంతఃవరుస uintptr_t util_sparse_array_కణుపు(శూన్య * దత్తాంశం, గుర్తులేని level) {
    ప్రతిజ్ఞా(దత్తాంశం != ఖాళీ);
    ప్రతిజ్ఞా(((uintptr_t) దత్తాంశం & కణుపు_LEVEL_ముసుగు) == 0);
    ప్రతిజ్ఞా((level & కణుపు_PTR_ముసుగు) == 0);
    తిరిగి (uintptr_t) దత్తాంశం | level;
}

అంతఃవరుస uintptr_t util_sparse_array_కణుపు_alloc(util_sparse_array * arr, గుర్తులేని level) {
    size_t size;
    ఒకవేళ (level == 0) {
        size = arr->elem_కొలత << arr->కణుపు_కొలత_log2;
    } లేకపోతే {
        size = పరిమాణం(uintptr_t) << arr->కణుపు_కొలత_log2;
    }

    శూన్య * దత్తాంశం = os_malloc_aligned(size, కణుపు_ALLOC_ALIGN);
    memset(దత్తాంశం, 0, size);

    తిరిగి util_sparse_array_కణుపు(దత్తాంశం, level);
}

స్థిర అంతఃవరుస uintptr_t util_sparse_array_సమితి_or_free_కణుపు(uintptr_t * కణుపు_ptr, uintptr_t cmp_కణుపు, uintptr_t కణుపు) {
    uintptr_t prev_కణుపు = p_atomic_cmpxchg(కణుపు_ptr, cmp_కణుపు, కణుపు);

    ఒకవేళ (prev_కణుపు != cmp_కణుపు) {
        /* We lost the race.  Free this one and return the one that was already
       * allocated.
       */
        os_free_aligned(util_sparse_array_కణుపు_దత్తాంశం(కణుపు));
        తిరిగి prev_కణుపు;
    } లేకపోతే {
        తిరిగి కణుపు;
    }
}

శూన్య * util_sparse_array_get(util_sparse_array * arr, uint64_t idx) {
    స్థిరరాశి గుర్తులేని కణుపు_కొలత_log2 = arr->కణుపు_కొలత_log2;
    uintptr_t      మూలం           = p_atomic_read(&arr->మూలం);
    ఒకవేళ (unlikely(!మూలం)) {
        గుర్తులేని మూలం_level = 0;
        uint64_t idx_iter   = idx >> కణుపు_కొలత_log2;
        అయితే (idx_iter) {
            idx_iter >>= కణుపు_కొలత_log2;
            మూలం_level++;
        }
        uintptr_t new_మూలం = util_sparse_array_కణుపు_alloc(arr, మూలం_level);
        మూలం               = util_sparse_array_సమితి_or_free_కణుపు(&arr->మూలం, NULL_కణుపు, new_మూలం);
    }

    అయితే (1) {
        గుర్తులేని మూలం_level = util_sparse_array_కణుపు_level(మూలం);
        uint64_t మూలం_idx   = idx >> (మూలం_level * కణుపు_కొలత_log2);
        ఒకవేళ (likely(మూలం_idx < (1ull << కణుపు_కొలత_log2))) {
            ఆపు;
        }

        /* In this case, we have a root but its level is low enough that the
       * requested index is out-of-bounds.
       */
        uintptr_t new_మూలం = util_sparse_array_కణుపు_alloc(arr, మూలం_level + 1);

        uintptr_t * new_మూలం_children = (uintptr_t *) util_sparse_array_కణుపు_దత్తాంశం(new_మూలం);
        new_మూలం_children[0]          = మూలం;

        /* We only add one at a time instead of the whole tree because it's
       * easier to ensure correctness of both the tree building and the
       * clean-up path.  Because we're only adding one node we never have to
       * worry about trying to free multiple things without freeing the old
       * things.
       */
        మూలం = util_sparse_array_సమితి_or_free_కణుపు(&arr->మూలం, మూలం, new_మూలం);
    }

    శూన్య *   కణుపు_దత్తాంశం  = util_sparse_array_కణుపు_దత్తాంశం(మూలం);
    గుర్తులేని కణుపు_level = util_sparse_array_కణుపు_level(మూలం);
    అయితే (కణుపు_level > 0) {
        uint64_t బిడ్డ_idx = (idx >> (కణుపు_level * కణుపు_కొలత_log2)) & ((1ull << కణుపు_కొలత_log2) - 1);

        uintptr_t * children = (uintptr_t *) కణుపు_దత్తాంశం;
        uintptr_t   బిడ్డ    = p_atomic_read(&children[బిడ్డ_idx]);

        ఒకవేళ (unlikely(!బిడ్డ)) {
            బిడ్డ = util_sparse_array_కణుపు_alloc(arr, కణుపు_level - 1);
            బిడ్డ = util_sparse_array_సమితి_or_free_కణుపు(&children[బిడ్డ_idx], NULL_కణుపు, బిడ్డ);
        }

        కణుపు_దత్తాంశం  = util_sparse_array_కణుపు_దత్తాంశం(బిడ్డ);
        కణుపు_level = util_sparse_array_కణుపు_level(బిడ్డ);
    }

    uint64_t elem_idx = idx & ((1ull << కణుపు_కొలత_log2) - 1);
    తిరిగి (శూన్య *) ((అక్షరం *) కణుపు_దత్తాంశం + (elem_idx * arr->elem_కొలత));
}
