#pragma once

// GGML internal header

#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "gguf.h"

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ifdef __ARM_FEATURE_SVE
#ಅನ್ತರ್ಭಾವ <arm_sve.h>
#endif // __ARM_FEATURE_SVE

#ಯದಿ defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#ಅನ್ತರ್ಭಾವ <arm_neon.h>
#endif

#ifdef __cplusplus
ಬಾಹ್ಯ "C" {
#endif

ಶೂನ್ಯ ggml_print_backtrace(ಶೂನ್ಯ);

uint64_t ggml_graph_next_uid(ಶೂನ್ಯ);

#ifndef MIN
#    define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#    define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

// required for mmap as gguf only guarantees 32-byte alignment
#define TENSOR_ALIGNMENT 32

// static_assert should be a #define, but if it's not,
// fall back to the _Static_assert C11 keyword.
// if C99 - static_assert is noop
// ref: https://stackoverflow.com/a/53923785/4039976
#ifndef __cplusplus
    #ifndef static_assert
        #ಯದಿ defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
            #define static_assert(cond, ಸಂದೇಶ) _Static_assert(cond, ಸಂದೇಶ)
        #ಅನ್ಯಥಾ
            #define static_assert(cond, ಸಂದೇಶ) ಸಂರಚನಾ global_scope_noop_trick
        #endif
    #endif
#endif

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ ggml_up32(ಪೂರ್ಣಾಙ್ಕ n) {
    ಪ್ರತ್ಯಾಗಮ (n + 31) & ~31;
}

//static inline int ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ ggml_up(ಪೂರ್ಣಾಙ್ಕ n, ಪೂರ್ಣಾಙ್ಕ m) {
    // assert m is a power of 2
    GGML_ASSERT((m & (m - 1)) == 0);
    ಪ್ರತ್ಯಾಗಮ (n + m - 1) & ~(m - 1);
}

// TODO: move to ggml.h? (won't be able to inline)
ಸ್ಥಿರ bool ggml_are_same_layout(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * a, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * b) {
    ಯದಿ (a->type != b->type) {
        ಪ್ರತ್ಯಾಗಮ false;
    }
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < GGML_MAX_DIMS; i++) {
        ಯದಿ (a->ne[i] != b->ne[i]) {
            ಪ್ರತ್ಯಾಗಮ false;
        }
        ಯದಿ (a->nb[i] != b->nb[i]) {
            ಪ್ರತ್ಯಾಗಮ false;
        }
    }
    ಪ್ರತ್ಯಾಗಮ true;
}

ಸ್ಥಿರ bool ggml_op_is_empty(ಗಣನಾ ggml_op op) {
    ವಿಕಲ್ಪ (op) {
        ಪಕ್ಷ GGML_OP_NONE:
        ಪಕ್ಷ GGML_OP_RESHAPE:
        ಪಕ್ಷ GGML_OP_TRANSPOSE:
        ಪಕ್ಷ GGML_OP_VIEW:
        ಪಕ್ಷ GGML_OP_PERMUTE:
            ಪ್ರತ್ಯಾಗಮ true;
        ಸ್ವಭಾವ:
            ಪ್ರತ್ಯಾಗಮ false;
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool ggml_impl_is_view(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t) {
    ಪ್ರತ್ಯಾಗಮ t->view_src != ರಿಕ್ತ;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ ggml_compute_softplus_f32(ಭಿನ್ನ ಒಳಹರಿವು) {
    ಪ್ರತ್ಯಾಗಮ (ಒಳಹರಿವು > 20.0f) ? ಒಳಹರಿವು : logf(1 + expf(ಒಳಹರಿವು));
}
//
// logging
//

GGML_ATTRIBUTE_FORMAT(2, 3)
GGML_API ಶೂನ್ಯ ggml_log_internal        (ಗಣನಾ ggml_log_level level, ಧ್ರುವ ವರ್ಣ * format, ...);
GGML_API ಶೂನ್ಯ ggml_log_callback_default(ಗಣನಾ ggml_log_level level, ಧ್ರುವ ವರ್ಣ * text, ಶೂನ್ಯ * user_data);

#define GGML_LOG(...)       ggml_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define GGML_LOG_INFO(...)  ggml_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define GGML_LOG_WARN(...)  ggml_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define GGML_LOG_ERROR(...) ggml_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GGML_LOG_DEBUG(...) ggml_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define GGML_LOG_CONT(...)  ggml_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

#define GGML_DEBUG 0

#ಯದಿ (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#ಅನ್ಯಥಾ
#define GGML_PRINT_DEBUG(...)
#endif

#ಯದಿ (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) GGML_LOG_DEBUG(__VA_ARGS__)
#ಅನ್ಯಥಾ
#define GGML_PRINT_DEBUG_5(...)
#endif

#ಯದಿ (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) GGML_LOG_DEBUG(__VA_ARGS__)
#ಅನ್ಯಥಾ
#define GGML_PRINT_DEBUG_10(...)
#endif

// tensor params

ಸ್ಥಿರ ಶೂನ್ಯ ggml_set_op_params(ಸಂರಚನಾ ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * params, size_t params_size) {
    GGML_ASSERT(tensor != ರಿಕ್ತ); // silence -Warray-bounds warnings
    ಪ್ರತಿಜ್ಞಾ(params_size <= GGML_MAX_OP_PARAMS);
    memcpy(tensor->op_params, params, params_size);
}

ಸ್ಥಿರ int32_t ggml_get_op_params_i32(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor, uint32_t i) {
    ಪ್ರತಿಜ್ಞಾ(i < GGML_MAX_OP_PARAMS / ಪರಿಮಾಣ(int32_t));
    ಪ್ರತ್ಯಾಗಮ ((ಧ್ರುವ int32_t *)(tensor->op_params))[i];
}

ಸ್ಥಿರ ಭಿನ್ನ ggml_get_op_params_f32(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor, uint32_t i) {
    ಪ್ರತಿಜ್ಞಾ(i < GGML_MAX_OP_PARAMS / ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಪ್ರತ್ಯಾಗಮ ((ಧ್ರುವ ಭಿನ್ನ *)(tensor->op_params))[i];
}

// [TAG_GGML_PREC]
// - GGML_OP_MUL_MAT
//   0 - acc
//   1 - hint
//   2 - src0 precision
//   3 - src1 precision
//
// - GGML_OP_MUL_MAT_ID
//   0 - acc
//   1 - hint
//   2 - src0 precision
//   3 - src1 precision
ಸ್ಥಿರ ಶೂನ್ಯ ggml_set_op_params_i32(ಸಂರಚನಾ ggml_tensor * tensor, uint32_t i, int32_t ಮೌಲ್ಯ) {
    ಪ್ರತಿಜ್ಞಾ(i < GGML_MAX_OP_PARAMS / ಪರಿಮಾಣ(int32_t));
    ((int32_t *)(tensor->op_params))[i] = ಮೌಲ್ಯ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_set_op_params_f32(ಸಂರಚನಾ ggml_tensor * tensor, uint32_t i, ಭಿನ್ನ ಮೌಲ್ಯ) {
    ಪ್ರತಿಜ್ಞಾ(i < GGML_MAX_OP_PARAMS / ಪರಿಮಾಣ(ಭಿನ್ನ));
    ((ಭಿನ್ನ *)(tensor->op_params))[i] = ಮೌಲ್ಯ;
}

ಸಂರಚನಾ ggml_map_custom1_op_params {
    ggml_custom1_op_t  fun;
    ಪೂರ್ಣಾಙ್ಕ                n_tasks;
    ಶೂನ್ಯ             * userdata;
};

ಸಂರಚನಾ ggml_map_custom2_op_params {
    ggml_custom2_op_t   fun;
    ಪೂರ್ಣಾಙ್ಕ                 n_tasks;
    ಶೂನ್ಯ              * userdata;
};

ಸಂರಚನಾ ggml_map_custom3_op_params {
    ggml_custom3_op_t fun;
    ಪೂರ್ಣಾಙ್ಕ               n_tasks;
    ಶೂನ್ಯ            * userdata;
};

ಸಂರಚನಾ ggml_custom_op_params {
    ggml_custom_op_t fun;
    ಪೂರ್ಣಾಙ್ಕ              n_tasks;
    ಶೂನ್ಯ           * userdata;
};

// bitset

ಪ್ರಕಾರನಿರ್ದೇಶ uint32_t ggml_bitset_t;

static_assert(ಪರಿಮಾಣ(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#define BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#define BITSET_MASK (ಪರಿಮಾಣ(ggml_bitset_t)*8 - 1)

ಸ್ಥಿರ size_t ggml_bitset_size(size_t n) {
    ಪ್ರತ್ಯಾಗಮ (n + BITSET_MASK) >> BITSET_SHR;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool ggml_bitset_get(ಧ್ರುವ ggml_bitset_t * bitset, size_t i) {
    ಪ್ರತ್ಯಾಗಮ !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_bitset_set(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_MASK));
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_bitset_clear(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_MASK));
}

// hash set

#define GGML_HASHSET_FULL ((size_t)-1)
#define GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)

ಸಂರಚನಾ ggml_hash_set {
    size_t size;
    ggml_bitset_t * used;       // whether or not the keys are in use i.e. set
    ಸಂರಚನಾ ggml_tensor ** ಕೀಲಿಗಳು; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};

ಸಂರಚನಾ ggml_hash_set ggml_hash_set_new(size_t size);
ಶೂನ್ಯ                 ggml_hash_set_free(ಸಂರಚನಾ ggml_hash_set * hash_set);

// returns the minimum size for a hash set that can hold min_sz elements
size_t ggml_hash_size(size_t min_sz);

// remove all elements from the hash set
ಶೂನ್ಯ ggml_hash_set_reset(ಸಂರಚನಾ ggml_hash_set * hash_set);

// returns true if key is in the hash set
ಸ್ಥಿರ bool ggml_hash_contains(ಧ್ರುವ ಸಂರಚನಾ ggml_hash_set * hash_set, ಸಂರಚನಾ ggml_tensor * ಕೀಲಿ);

// returns GGML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
ಸ್ಥಿರ size_t ggml_hash_find(ಧ್ರುವ ಸಂರಚನಾ ggml_hash_set * hash_set, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * ಕೀಲಿ);

// returns GGML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
ಸ್ಥಿರ size_t ggml_hash_ಸೇರಿಸು(ಸಂರಚನಾ ggml_hash_set * hash_set, ಸಂರಚನಾ ggml_tensor * ಕೀಲಿ);

// return index, asserts if table is full
ಸ್ಥಿರ size_t ggml_hash_find_or_ಸೇರಿಸು(ಸಂರಚನಾ ggml_hash_set * hash_set, ಸಂರಚನಾ ggml_tensor * ಕೀಲಿ);

// hash function for ggml_tensor
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t ggml_hash(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    ಪ್ರತ್ಯಾಗಮ (size_t)(uintptr_t)p >> 4;
}

ಸ್ಥಿರ size_t ggml_hash_find(ಧ್ರುವ ಸಂರಚನಾ ggml_hash_set * hash_set, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * ಕೀಲಿ) {
    size_t h = ggml_hash(ಕೀಲಿ) % hash_set->size;

    // linear probing
    size_t i = h;
    ಯಾವತ್ (ggml_bitset_get(hash_set->used, i) && hash_set->ಕೀಲಿಗಳು[i] != ಕೀಲಿ) {
        i = (i + 1) % hash_set->size;
        ಯದಿ (i == h) {
            // visited all hash table entries -> not found
            ಪ್ರತ್ಯಾಗಮ GGML_HASHSET_FULL;
        }
    }
    ಪ್ರತ್ಯಾಗಮ i;
}

ಸ್ಥಿರ bool ggml_hash_contains(ಧ್ರುವ ಸಂರಚನಾ ggml_hash_set * hash_set, ಸಂರಚನಾ ggml_tensor * ಕೀಲಿ) {
    size_t i = ggml_hash_find(hash_set, ಕೀಲಿ);
    ಪ್ರತ್ಯಾಗಮ i != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, i);
}

ಸ್ಥಿರ size_t ggml_hash_ಸೇರಿಸು(ಸಂರಚನಾ ggml_hash_set * hash_set, ಸಂರಚನಾ ggml_tensor * ಕೀಲಿ) {
    size_t h = ggml_hash(ಕೀಲಿ) % hash_set->size;

    // linear probing
    size_t i = h;
    ಕುರು {
        ಯದಿ (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->ಕೀಲಿಗಳು[i] = ಕೀಲಿ;
            ಪ್ರತ್ಯಾಗಮ i;
        }
        ಯದಿ (hash_set->ಕೀಲಿಗಳು[i] == ಕೀಲಿ) {
            ಪ್ರತ್ಯಾಗಮ GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_set->size;
    } ಯಾವತ್ (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

ಸ್ಥಿರ size_t ggml_hash_find_or_ಸೇರಿಸು(ಸಂರಚನಾ ggml_hash_set * hash_set, ಸಂರಚನಾ ggml_tensor * ಕೀಲಿ) {
    size_t h = ggml_hash(ಕೀಲಿ) % hash_set->size;

    // linear probing
    size_t i = h;
    ಕುರು {
        ಯದಿ (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->ಕೀಲಿಗಳು[i] = ಕೀಲಿ;
            ಪ್ರತ್ಯಾಗಮ i;
        }
        ಯದಿ (hash_set->ಕೀಲಿಗಳು[i] == ಕೀಲಿ) {
            ಪ್ರತ್ಯಾಗಮ i;
        }
        i = (i + 1) % hash_set->size;
    } ಯಾವತ್ (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

// computation graph

ಗಣನಾ ggml_cgraph_eval_order {
    GGML_CGRAPH_EVAL_ORDER_ಎಡ_TO_ಬಲ = 0,
    GGML_CGRAPH_EVAL_ORDER_ಬಲ_TO_ಎಡ,
    GGML_CGRAPH_EVAL_ORDER_ಎಣಿಕೆ
};

ಸಂರಚನಾ ggml_cgraph {
    ಪೂರ್ಣಾಙ್ಕ size;    // maximum number of nodes/leafs/grads/grad_accs
    ಪೂರ್ಣಾಙ್ಕ n_nodes; // number of nodes currently in use
    ಪೂರ್ಣಾಙ್ಕ n_leafs; // number of leafs currently in use

    ಸಂರಚನಾ ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
    ಸಂರಚನಾ ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
    ಸಂರಚನಾ ggml_tensor ** grad_accs; // accumulators for node gradients
    ಸಂರಚನಾ ggml_tensor ** leafs;     // tensors with constant data
    int32_t             * use_counts;// number of uses of each tensor, indexed by hash table slot

    ಸಂರಚನಾ ggml_hash_set visited_hash_set;

    ಗಣನಾ ggml_cgraph_eval_order order;

    // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
    // a value of 0 means it is not set and should be ignored
    uint64_t uid;
};

// returns a slice of cgraph with nodes [i0, i1)
// the slice does not have leafs or gradients
// if you need the gradients, get them from the original graph
ಸಂರಚನಾ ggml_cgraph ggml_graph_view(ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ i0, ಪೂರ್ಣಾಙ್ಕ i1);

// ggml-alloc.c: true if the operation can reuse memory from its sources
GGML_API bool ggml_op_can_inplace(ಗಣನಾ ggml_op op);


// Memory allocation

GGML_API ಶೂನ್ಯ * ggml_aligned_malloc(size_t size);
GGML_API ಶೂನ್ಯ ggml_aligned_free(ಶೂನ್ಯ * ptr, size_t size);

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ fp32_from_bits(uint32_t w) {
    ಸಙ್ಘ {
        uint32_t as_bits;
        ಭಿನ್ನ as_ಮೌಲ್ಯ;
    } fp32;
    fp32.as_bits = w;
    ಪ್ರತ್ಯಾಗಮ fp32.as_ಮೌಲ್ಯ;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint32_t fp32_to_bits(ಭಿನ್ನ f) {
    ಸಙ್ಘ {
        ಭಿನ್ನ as_ಮೌಲ್ಯ;
        uint32_t as_bits;
    } fp32;
    fp32.as_ಮೌಲ್ಯ = f;
    ಪ್ರತ್ಯಾಗಮ fp32.as_bits;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    ಧ್ರುವ uint32_t w = (uint32_t) h << 16;
    ಧ್ರುವ uint32_t sign = w & UINT32_C(0x80000000);
    ಧ್ರುವ uint32_t two_w = w + w;

    ಧ್ರುವ uint32_t exp_offset = UINT32_C(0xE0) << 23;
#ಯದಿ (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    ಧ್ರುವ ಭಿನ್ನ exp_scale = 0x1.0p-112f;
#ಅನ್ಯಥಾ
    ಧ್ರುವ ಭಿನ್ನ exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    ಧ್ರುವ ಭಿನ್ನ normalized_ಮೌಲ್ಯ = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    ಧ್ರುವ uint32_t magic_mask = UINT32_C(126) << 23;
    ಧ್ರುವ ಭಿನ್ನ magic_bias = 0.5f;
    ಧ್ರುವ ಭಿನ್ನ denormalized_ಮೌಲ್ಯ = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    ಧ್ರುವ uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    ಧ್ರುವ uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_ಮೌಲ್ಯ) : fp32_to_bits(normalized_ಮೌಲ್ಯ));
    ಪ್ರತ್ಯಾಗಮ fp32_from_bits(result);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ggml_fp16_t ggml_compute_fp32_to_fp16(ಭಿನ್ನ f) {
#ಯದಿ (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    ಧ್ರುವ ಭಿನ್ನ scale_to_inf = 0x1.0p+112f;
    ಧ್ರುವ ಭಿನ್ನ scale_to_zero = 0x1.0p-110f;
#ಅನ್ಯಥಾ
    ಧ್ರುವ ಭಿನ್ನ scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    ಧ್ರುವ ಭಿನ್ನ scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    ಭಿನ್ನ base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    ಧ್ರುವ uint32_t w = fp32_to_bits(f);
    ಧ್ರುವ uint32_t shl1_w = w + w;
    ಧ್ರುವ uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    ಯದಿ (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    ಧ್ರುವ uint32_t bits = fp32_to_bits(base);
    ಧ್ರುವ uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    ಧ್ರುವ uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    ಧ್ರುವ uint32_t nonsign = exp_bits + mantissa_bits;
    ಪ್ರತ್ಯಾಗಮ (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ ggml_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;  // Stores the raw bit representation of the float

    // Handle special case for minimum exponent (denormalized float)
    ಯದಿ (x == 0) {
        // Bit pattern for 2^(-127):
        // - Sign bit: 0 (positive)
        // - Exponent: 0 (denormalized number)
        // - Mantissa: 0x400000 (0.5 in fractional form)
        // Value = 0.5 * 2^(-126) = 2^(-127)
        bits = 0x00400000;
    }
    // note: disabled as we don't need to handle NaNs
    //// Handle special case for NaN (all bits set)
    //else if (x == 0xFF) {
    //    // Standard quiet NaN pattern:
    //    // - Sign bit: 0
    //    // - Exponent: all 1s (0xFF)
    //    // - Mantissa: 0x400000 (quiet NaN flag)
    //    bits = 0x7FC00000;
    //}
    // Normalized values (most common case)
    ಅನ್ಯಥಾ {
        // Construct normalized float by shifting exponent into position:
        // - Exponent field: 8 bits (positions 30-23)
        // - Mantissa: 0 (implicit leading 1)
        // Value = 2^(x - 127)
        bits = (uint32_t) x << 23;
    }

    ಭಿನ್ನ result;  // Final float value
                   // Safely reinterpret bit pattern as float without type-punning issues
    memcpy(&result, &bits, ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಪ್ರತ್ಯಾಗಮ result;
}

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ ggml_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;

    // For x < 2: use precomputed denormal patterns
    ಯದಿ (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        bits = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    ಅನ್ಯಥಾ {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        bits = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    ಭಿನ್ನ result;
    memcpy(&result, &bits, ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಪ್ರತ್ಯಾಗಮ result;
}

#define GGML_E8M0_TO_FP32(x) ggml_e8m0_to_fp32(x)
#define GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ ggml_ue4m3_to_fp32(uint8_t x) {
    ಯದಿ (x == 0 || x == 0x7F) {
        ಪ್ರತ್ಯಾಗಮ 0.0f;
    }
    ಪೂರ್ಣಾಙ್ಕ   exp = (x >> 3) & 0xF;
    ಪೂರ್ಣಾಙ್ಕ   man = x & 0x7;
    ಭಿನ್ನ raw;
    ಯದಿ (exp == 0) {
        raw = ldexpf((ಭಿನ್ನ) man, -9);
    } ಅನ್ಯಥಾ {
        raw = ldexpf(1.0f + (ಭಿನ್ನ) man / 8.0f, exp - 7);
    }
    ಪ್ರತ್ಯಾಗಮ raw * 0.5f;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint8_t ggml_fp32_to_ue4m3(ಭಿನ್ನ x) {
    ಯದಿ (!(x > 0.0f)) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }
    ಯದಿ (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t bits;
    memcpy(&bits, &x, 4);
    ಪೂರ್ಣಾಙ್ಕ fp32_exp  = ((bits >> 23) & 0xFF) - 127;
    ಪೂರ್ಣಾಙ್ಕ fp32_man  = (bits >> 20) & 0x7;
    ಪೂರ್ಣಾಙ್ಕ ue4m3_exp = fp32_exp + 7;
    ಯದಿ (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        ಪೂರ್ಣಾಙ್ಕ man = (ಪೂರ್ಣಾಙ್ಕ) (x * 512.0f + 0.5f);
        ಯದಿ (man > 7) {
            man = 7;
        }
        ಯದಿ (man < 1) {
            ಪ್ರತ್ಯಾಗಮ 0;
        }
        ಪ್ರತ್ಯಾಗಮ (uint8_t) man;
    }
    ಯದಿ (ue4m3_exp >= 15) {
        ಪ್ರತ್ಯಾಗಮ 0x7E;
    }
    ಪೂರ್ಣಾಙ್ಕ round_bit = (bits >> 19) & 1;
    ಪೂರ್ಣಾಙ್ಕ ue4m3_man = fp32_man + round_bit;
    ಯದಿ (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        ಯದಿ (ue4m3_exp >= 15) {
            ಪ್ರತ್ಯಾಗಮ 0x7E;
        }
    }
    ಪ್ರತ್ಯಾಗಮ (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
}

/**
 * Converts brain16 to float32.
 *
 * The bfloat16 floating point format has the following structure:
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───┐
 *     0b0000000000000000 brain16
 *
 * Since bf16 has the same number of exponent bits as a 32bit float,
 * encoding and decoding numbers becomes relatively straightforward.
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───────────────────┐
 *     0b00000000000000000000000000000000 IEEE binary32
 *
 * For comparison, the standard fp16 format has fewer exponent bits.
 *
 *       ┌sign
 *       │
 *       │  ┌exponent
 *       │  │
 *       │  │    ┌mantissa
 *       │  │    │
 *       │┌─┴─┐┌─┴──────┐
 *     0b0000000000000000 IEEE binary16
 *
 * @see IEEE 754-2008
 */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    ಸಙ್ಘ {
        ಭಿನ್ನ f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    ಪ್ರತ್ಯಾಗಮ u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ggml_bf16_t ggml_compute_fp32_to_bf16(ಭಿನ್ನ s) {
    ggml_bf16_t h;
    ಸಙ್ಘ {
        ಭಿನ್ನ f;
        uint32_t i;
    } u;
    u.f = s;
    ಯದಿ ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.bits = (u.i >> 16) | 64; /* force to quiet */
        ಪ್ರತ್ಯಾಗಮ h;
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    ಪ್ರತ್ಯಾಗಮ h;
}

#define GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)
#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ int32_t ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * ಗಂಟು = cgraph->nodes[ಗಂಟು_idx];

    size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, ಗಂಟು);
    ಯದಿ (!ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }
    ಪ್ರತ್ಯಾಗಮ cgraph->use_counts[hash_pos];
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool ggml_ಗಂಟು_has_n_uses(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, int32_t n_uses) {
    ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * ಗಂಟು = cgraph->nodes[ಗಂಟು_idx];

    // check the use count against how many we're replacing
    ಯದಿ (ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, ಗಂಟು_idx) != n_uses) {
        ಪ್ರತ್ಯಾಗಮ false;
    }

    // if node is a view, some other node might be using the intermediate result
    // via the view source.
    ಯದಿ (ಗಂಟು->view_src) {
        ಪ್ರತ್ಯಾಗಮ false;
    }

    // If the user requested output for the node, can't fuse
    ಯದಿ (ಗಂಟು->flags & GGML_TENSOR_FLAG_OUTPUT) {
        ಪ್ರತ್ಯಾಗಮ false;
    }

    ಪ್ರತ್ಯಾಗಮ true;
}

// Returns true if nodes with indices { node_idxs } are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool ggml_can_fuse_ext(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * ಗಂಟು_idxs, ಧ್ರುವ ಗಣನಾ ggml_op * ops, ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_ops) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ಅಂಕೆ_ops; ++i) {
        ಯದಿ (ಗಂಟು_idxs[i] >= cgraph->n_nodes) {
            ಪ್ರತ್ಯಾಗಮ false;
        }

        ಸಂರಚನಾ ggml_tensor * ಗಂಟು = cgraph->nodes[ಗಂಟು_idxs[i]];
        ಯದಿ (ಗಂಟು->op != ops[i]) {
            ಪ್ರತ್ಯಾಗಮ false;
        }
        ಯದಿ ((ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ಪ್ರತ್ಯಾಗಮ false;
        }
        ಯದಿ (i < ಅಂಕೆ_ops - 1 && !ggml_ಗಂಟು_has_n_uses(cgraph, ಗಂಟು_idxs[i], 1)) {
            ಪ್ರತ್ಯಾಗಮ false;
        }
        ಯದಿ (i > 0) {
            ಸಂರಚನಾ ggml_tensor * prev = cgraph->nodes[ಗಂಟು_idxs[i - 1]];
            ಯದಿ (ಗಂಟು->src[0] != prev && ಗಂಟು->src[1] != prev) {
                ಪ್ರತ್ಯಾಗಮ false;
            }
            ಯದಿ (!ggml_are_same_shape(ಗಂಟು, prev)) {
                ಪ್ರತ್ಯಾಗಮ false;
            }
        }
    }
    ಪ್ರತ್ಯಾಗಮ true;
}

// same as above, for sequential indices starting at node_idx
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool ggml_can_fuse(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, ಧ್ರುವ ಗಣನಾ ggml_op * ops, ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_ops) {
    ಪ್ರತಿಜ್ಞಾ(ಅಂಕೆ_ops < 32);

    ಯದಿ (ಗಂಟು_idx + ಅಂಕೆ_ops > cgraph->n_nodes) {
        ಪ್ರತ್ಯಾಗಮ false;
    }

    ಪೂರ್ಣಾಙ್ಕ idxs[32];
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ಅಂಕೆ_ops; ++i) {
        idxs[i] = ಗಂಟು_idx + i;
    }

    ಪ್ರತ್ಯಾಗಮ ggml_can_fuse_ext(cgraph, idxs, ops, ಅಂಕೆ_ops);
}

GGML_API bool ggml_can_fuse_subgraph_ext(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph,
                                         ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *                ಗಂಟು_idxs,
                                         ಪೂರ್ಣಾಙ್ಕ                        ಎಣಿಕೆ,
                                         ಧ್ರುವ ಗಣನಾ ggml_op *       ops,
                                         ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *                outputs,
                                         ಪೂರ್ಣಾಙ್ಕ                        ಅಂಕೆ_outputs);

// Returns true if the subgraph formed by {node_idxs} can be fused
// checks whethers all nodes which are not part of outputs can be elided
// by checking if their num_uses are confined to the subgraph
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ bool ggml_can_fuse_subgraph(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph,
                                          ಪೂರ್ಣಾಙ್ಕ                        ಗಂಟು_idx,
                                          ಪೂರ್ಣಾಙ್ಕ                        ಎಣಿಕೆ,
                                          ಧ್ರುವ ಗಣನಾ ggml_op *       ops,
                                          ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *                outputs,
                                          ಪೂರ್ಣಾಙ್ಕ                        ಅಂಕೆ_outputs) {
    GGML_ASSERT(ಎಣಿಕೆ < 32);
    ಯದಿ (ಗಂಟು_idx + ಎಣಿಕೆ > cgraph->n_nodes) {
        ಪ್ರತ್ಯಾಗಮ false;
    }

    ಪೂರ್ಣಾಙ್ಕ idxs[32];

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ಎಣಿಕೆ; ++i) {
        idxs[i] = ಗಂಟು_idx + i;
    }

    ಪ್ರತ್ಯಾಗಮ ggml_can_fuse_subgraph_ext(cgraph, idxs, ಎಣಿಕೆ, ops, outputs, ಅಂಕೆ_outputs);
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#ಅನ್ತರ್ಭಾವ <array>
#ಅನ್ತರ್ಭಾವ <initializer_list>
#ಅನ್ತರ್ಭಾವ <vector>

// nicer C++ syntax for ggml_can_fuse
ಅನ್ತಃಸ್ಥ bool ggml_can_fuse(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, std::initializer_list<ಗಣನಾ ggml_op> ops) {
    ಪ್ರತ್ಯಾಗಮ ggml_can_fuse(cgraph, ಗಂಟು_idx, ops.begin(), (ಪೂರ್ಣಾಙ್ಕ)ops.size());
}

ಅನ್ತಃಸ್ಥ bool ggml_can_fuse_subgraph(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph *          cgraph,
                                   ಪೂರ್ಣಾಙ್ಕ                                 start_idx,
                                   std::initializer_list<ಗಣನಾ ggml_op> ops,
                                   std::initializer_list<ಪೂರ್ಣಾಙ್ಕ>          outputs = {}) {
    ಪ್ರತ್ಯಾಗಮ ggml_can_fuse_subgraph(cgraph, start_idx, ops.size(), ops.begin(), outputs.begin(), outputs.size());
}

// Return true if the edges in the graph match expectations.
ಅನ್ತಃಸ್ಥ bool ggml_check_edges(ಧ್ರುವ ಸಂರಚನಾ ggml_cgraph *                cgraph,
                             ಪೂರ್ಣಾಙ್ಕ                                       start_idx,
                             std::initializer_list<std::array<ಪೂರ್ಣಾಙ್ಕ, 3>> edges) {
    ಪ್ರತಿ (ಧ್ರುವ auto & edge : edges) {
        ಪೂರ್ಣಾಙ್ಕ dst_ಗಂಟು = edge[0];
        ಪೂರ್ಣಾಙ್ಕ src_idx  = edge[1];
        ಪೂರ್ಣಾಙ್ಕ src_ಗಂಟು = edge[2];
        ಯದಿ (cgraph->nodes[start_idx + dst_ಗಂಟು]->src[src_idx] != cgraph->nodes[start_idx + src_ಗಂಟು]) {
            ಪ್ರತ್ಯಾಗಮ false;
        }
    }
    ಪ್ರತ್ಯಾಗಮ true;
}

// expose GGUF internals for test code
GGML_API size_t gguf_type_size(ಗಣನಾ gguf_type type);
GGML_API ಶೂನ್ಯ gguf_write_to_buf(ಧ್ರುವ ಸಂರಚನಾ gguf_context * ctx, std::vector<int8_t> & buf, bool only_meta);
#endif // __cplusplus
