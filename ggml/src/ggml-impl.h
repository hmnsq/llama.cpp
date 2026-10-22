#pragma once

// GGML internal header

#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "gguf.h"

#ଅନ୍ତର୍ଭାଵ <assert.h>
#ଅନ୍ତର୍ଭାଵ <math.h>
#ଅନ୍ତର୍ଭାଵ <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#ଅନ୍ତର୍ଭାଵ <stdbool.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <string.h>

#ifdef __ARM_FEATURE_SVE
#ଅନ୍ତର୍ଭାଵ <arm_sve.h>
#endif // __ARM_FEATURE_SVE

#ଯଦି defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#ଅନ୍ତର୍ଭାଵ <arm_neon.h>
#endif

#ifdef __cplusplus
ବାହ୍ୟ "C" {
#endif

ଶୂନ୍ୟ ggml_print_backtrace(ଶୂନ୍ୟ);

uint64_t ggml_graph_next_uid(ଶୂନ୍ୟ);

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
        #ଯଦି defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
            #define static_assert(cond, ସନ୍ଦେଶ) _Static_assert(cond, ସନ୍ଦେଶ)
        #ଅନ୍ୟଥା
            #define static_assert(cond, ସନ୍ଦେଶ) ସଂରଚନା global_scope_noop_trick
        #endif
    #endif
#endif

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_up32(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n) {
    ଫେରନ୍ତୁ (n + 31) & ~31;
}

//static inline int ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_up(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ପୂର୍ଣ୍ଣସଂଖ୍ୟା m) {
    // assert m is a power of 2
    GGML_ASSERT((m & (m - 1)) == 0);
    ଫେରନ୍ତୁ (n + m - 1) & ~(m - 1);
}

// TODO: move to ggml.h? (won't be able to inline)
ସ୍ଥିର bool ggml_are_same_layout(ଅଚଳ ସଂରଚନା ggml_tensor * a, ଅଚଳ ସଂରଚନା ggml_tensor * b) {
    ଯଦି (a->type != b->type) {
        ଫେରନ୍ତୁ false;
    }
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < GGML_MAX_DIMS; i++) {
        ଯଦି (a->ne[i] != b->ne[i]) {
            ଫେରନ୍ତୁ false;
        }
        ଯଦି (a->nb[i] != b->nb[i]) {
            ଫେରନ୍ତୁ false;
        }
    }
    ଫେରନ୍ତୁ true;
}

ସ୍ଥିର bool ggml_op_is_empty(ଗଣନା ggml_op op) {
    ବାଛ (op) {
        ସ୍ଥିତି GGML_OP_NONE:
        ସ୍ଥିତି GGML_OP_RESHAPE:
        ସ୍ଥିତି GGML_OP_TRANSPOSE:
        ସ୍ଥିତି GGML_OP_VIEW:
        ସ୍ଥିତି GGML_OP_PERMUTE:
            ଫେରନ୍ତୁ true;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ false;
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool ggml_impl_is_view(ଅଚଳ ସଂରଚନା ggml_tensor * t) {
    ଫେରନ୍ତୁ t->view_src != ଖାଲି;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ ggml_compute_softplus_f32(ଦଶମିକ ନିବେଶ) {
    ଫେରନ୍ତୁ (ନିବେଶ > 20.0f) ? ନିବେଶ : logf(1 + expf(ନିବେଶ));
}
//
// logging
//

GGML_ATTRIBUTE_FORMAT(2, 3)
GGML_API ଶୂନ୍ୟ ggml_log_internal        (ଗଣନା ggml_log_level level, ଅଚଳ ଅକ୍ଷର * format, ...);
GGML_API ଶୂନ୍ୟ ggml_log_callback_default(ଗଣନା ggml_log_level level, ଅଚଳ ଅକ୍ଷର * text, ଶୂନ୍ୟ * user_data);

#define GGML_LOG(...)       ggml_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define GGML_LOG_INFO(...)  ggml_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define GGML_LOG_WARN(...)  ggml_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define GGML_LOG_ERROR(...) ggml_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GGML_LOG_DEBUG(...) ggml_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define GGML_LOG_CONT(...)  ggml_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

#define GGML_DEBUG 0

#ଯଦି (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#ଅନ୍ୟଥା
#define GGML_PRINT_DEBUG(...)
#endif

#ଯଦି (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) GGML_LOG_DEBUG(__VA_ARGS__)
#ଅନ୍ୟଥା
#define GGML_PRINT_DEBUG_5(...)
#endif

#ଯଦି (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) GGML_LOG_DEBUG(__VA_ARGS__)
#ଅନ୍ୟଥା
#define GGML_PRINT_DEBUG_10(...)
#endif

// tensor params

ସ୍ଥିର ଶୂନ୍ୟ ggml_set_op_params(ସଂରଚନା ggml_tensor * tensor, ଅଚଳ ଶୂନ୍ୟ * params, size_t params_size) {
    GGML_ASSERT(tensor != ଖାଲି); // silence -Warray-bounds warnings
    ପ୍ରତିଜ୍ଞା(params_size <= GGML_MAX_OP_PARAMS);
    memcpy(tensor->op_params, params, params_size);
}

ସ୍ଥିର int32_t ggml_get_op_params_i32(ଅଚଳ ସଂରଚନା ggml_tensor * tensor, uint32_t i) {
    ପ୍ରତିଜ୍ଞା(i < GGML_MAX_OP_PARAMS / ଆକାର(int32_t));
    ଫେରନ୍ତୁ ((ଅଚଳ int32_t *)(tensor->op_params))[i];
}

ସ୍ଥିର ଦଶମିକ ggml_get_op_params_f32(ଅଚଳ ସଂରଚନା ggml_tensor * tensor, uint32_t i) {
    ପ୍ରତିଜ୍ଞା(i < GGML_MAX_OP_PARAMS / ଆକାର(ଦଶମିକ));
    ଫେରନ୍ତୁ ((ଅଚଳ ଦଶମିକ *)(tensor->op_params))[i];
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
ସ୍ଥିର ଶୂନ୍ୟ ggml_set_op_params_i32(ସଂରଚନା ggml_tensor * tensor, uint32_t i, int32_t ମୂଲ୍ୟ) {
    ପ୍ରତିଜ୍ଞା(i < GGML_MAX_OP_PARAMS / ଆକାର(int32_t));
    ((int32_t *)(tensor->op_params))[i] = ମୂଲ୍ୟ;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_set_op_params_f32(ସଂରଚନା ggml_tensor * tensor, uint32_t i, ଦଶମିକ ମୂଲ୍ୟ) {
    ପ୍ରତିଜ୍ଞା(i < GGML_MAX_OP_PARAMS / ଆକାର(ଦଶମିକ));
    ((ଦଶମିକ *)(tensor->op_params))[i] = ମୂଲ୍ୟ;
}

ସଂରଚନା ggml_map_custom1_op_params {
    ggml_custom1_op_t  fun;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                n_tasks;
    ଶୂନ୍ୟ             * userdata;
};

ସଂରଚନା ggml_map_custom2_op_params {
    ggml_custom2_op_t   fun;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                 n_tasks;
    ଶୂନ୍ୟ              * userdata;
};

ସଂରଚନା ggml_map_custom3_op_params {
    ggml_custom3_op_t fun;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା               n_tasks;
    ଶୂନ୍ୟ            * userdata;
};

ସଂରଚନା ggml_custom_op_params {
    ggml_custom_op_t fun;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା              n_tasks;
    ଶୂନ୍ୟ           * userdata;
};

// bitset

ପ୍ରକାରବ୍ୟାଖ୍ୟା uint32_t ggml_bitset_t;

static_assert(ଆକାର(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#define BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#define BITSET_MASK (ଆକାର(ggml_bitset_t)*8 - 1)

ସ୍ଥିର size_t ggml_bitset_size(size_t n) {
    ଫେରନ୍ତୁ (n + BITSET_MASK) >> BITSET_SHR;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool ggml_bitset_get(ଅଚଳ ggml_bitset_t * bitset, size_t i) {
    ଫେରନ୍ତୁ !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_bitset_set(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_MASK));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_bitset_clear(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_MASK));
}

// hash set

#define GGML_HASHSET_FULL ((size_t)-1)
#define GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)

ସଂରଚନା ggml_hash_set {
    size_t size;
    ggml_bitset_t * used;       // whether or not the keys are in use i.e. set
    ସଂରଚନା ggml_tensor ** ଚାବିଗୁଡ଼ିକ; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};

ସଂରଚନା ggml_hash_set ggml_hash_set_new(size_t size);
ଶୂନ୍ୟ                 ggml_hash_set_free(ସଂରଚନା ggml_hash_set * hash_set);

// returns the minimum size for a hash set that can hold min_sz elements
size_t ggml_hash_size(size_t min_sz);

// remove all elements from the hash set
ଶୂନ୍ୟ ggml_hash_set_reset(ସଂରଚନା ggml_hash_set * hash_set);

// returns true if key is in the hash set
ସ୍ଥିର bool ggml_hash_contains(ଅଚଳ ସଂରଚନା ggml_hash_set * hash_set, ସଂରଚନା ggml_tensor * ଚାବି);

// returns GGML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
ସ୍ଥିର size_t ggml_hash_find(ଅଚଳ ସଂରଚନା ggml_hash_set * hash_set, ଅଚଳ ସଂରଚନା ggml_tensor * ଚାବି);

// returns GGML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
ସ୍ଥିର size_t ggml_hash_ଭର୍ତ୍ତି(ସଂରଚନା ggml_hash_set * hash_set, ସଂରଚନା ggml_tensor * ଚାବି);

// return index, asserts if table is full
ସ୍ଥିର size_t ggml_hash_find_or_ଭର୍ତ୍ତି(ସଂରଚନା ggml_hash_set * hash_set, ସଂରଚନା ggml_tensor * ଚାବି);

// hash function for ggml_tensor
ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t ggml_hash(ଅଚଳ ସଂରଚନା ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    ଫେରନ୍ତୁ (size_t)(uintptr_t)p >> 4;
}

ସ୍ଥିର size_t ggml_hash_find(ଅଚଳ ସଂରଚନା ggml_hash_set * hash_set, ଅଚଳ ସଂରଚନା ggml_tensor * ଚାବି) {
    size_t h = ggml_hash(ଚାବି) % hash_set->size;

    // linear probing
    size_t i = h;
    ଯେତେବେଳେ (ggml_bitset_get(hash_set->used, i) && hash_set->ଚାବିଗୁଡ଼ିକ[i] != ଚାବି) {
        i = (i + 1) % hash_set->size;
        ଯଦି (i == h) {
            // visited all hash table entries -> not found
            ଫେରନ୍ତୁ GGML_HASHSET_FULL;
        }
    }
    ଫେରନ୍ତୁ i;
}

ସ୍ଥିର bool ggml_hash_contains(ଅଚଳ ସଂରଚନା ggml_hash_set * hash_set, ସଂରଚନା ggml_tensor * ଚାବି) {
    size_t i = ggml_hash_find(hash_set, ଚାବି);
    ଫେରନ୍ତୁ i != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, i);
}

ସ୍ଥିର size_t ggml_hash_ଭର୍ତ୍ତି(ସଂରଚନା ggml_hash_set * hash_set, ସଂରଚନା ggml_tensor * ଚାବି) {
    size_t h = ggml_hash(ଚାବି) % hash_set->size;

    // linear probing
    size_t i = h;
    କର {
        ଯଦି (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->ଚାବିଗୁଡ଼ିକ[i] = ଚାବି;
            ଫେରନ୍ତୁ i;
        }
        ଯଦି (hash_set->ଚାବିଗୁଡ଼ିକ[i] == ଚାବି) {
            ଫେରନ୍ତୁ GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_set->size;
    } ଯେତେବେଳେ (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

ସ୍ଥିର size_t ggml_hash_find_or_ଭର୍ତ୍ତି(ସଂରଚନା ggml_hash_set * hash_set, ସଂରଚନା ggml_tensor * ଚାବି) {
    size_t h = ggml_hash(ଚାବି) % hash_set->size;

    // linear probing
    size_t i = h;
    କର {
        ଯଦି (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->ଚାବିଗୁଡ଼ିକ[i] = ଚାବି;
            ଫେରନ୍ତୁ i;
        }
        ଯଦି (hash_set->ଚାବିଗୁଡ଼ିକ[i] == ଚାବି) {
            ଫେରନ୍ତୁ i;
        }
        i = (i + 1) % hash_set->size;
    } ଯେତେବେଳେ (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

// computation graph

ଗଣନା ggml_cgraph_eval_order {
    GGML_CGRAPH_EVAL_ORDER_ବାମ_TO_ଡାହାଣ = 0,
    GGML_CGRAPH_EVAL_ORDER_ଡାହାଣ_TO_ବାମ,
    GGML_CGRAPH_EVAL_ORDER_ଗଣନା
};

ସଂରଚନା ggml_cgraph {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା size;    // maximum number of nodes/leafs/grads/grad_accs
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_nodes; // number of nodes currently in use
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_leafs; // number of leafs currently in use

    ସଂରଚନା ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
    ସଂରଚନା ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
    ସଂରଚନା ggml_tensor ** grad_accs; // accumulators for node gradients
    ସଂରଚନା ggml_tensor ** leafs;     // tensors with constant data
    int32_t             * use_counts;// number of uses of each tensor, indexed by hash table slot

    ସଂରଚନା ggml_hash_set visited_hash_set;

    ଗଣନା ggml_cgraph_eval_order order;

    // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
    // a value of 0 means it is not set and should be ignored
    uint64_t uid;
};

// returns a slice of cgraph with nodes [i0, i1)
// the slice does not have leafs or gradients
// if you need the gradients, get them from the original graph
ସଂରଚନା ggml_cgraph ggml_graph_view(ସଂରଚନା ggml_cgraph * cgraph, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା i1);

// ggml-alloc.c: true if the operation can reuse memory from its sources
GGML_API bool ggml_op_can_inplace(ଗଣନା ggml_op op);


// Memory allocation

GGML_API ଶୂନ୍ୟ * ggml_aligned_malloc(size_t size);
GGML_API ଶୂନ୍ୟ ggml_aligned_free(ଶୂନ୍ୟ * ptr, size_t size);

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ fp32_from_bits(uint32_t w) {
    ସଂଘ {
        uint32_t as_bits;
        ଦଶମିକ as_ମୂଲ୍ୟ;
    } fp32;
    fp32.as_bits = w;
    ଫେରନ୍ତୁ fp32.as_ମୂଲ୍ୟ;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint32_t fp32_to_bits(ଦଶମିକ f) {
    ସଂଘ {
        ଦଶମିକ as_ମୂଲ୍ୟ;
        uint32_t as_bits;
    } fp32;
    fp32.as_ମୂଲ୍ୟ = f;
    ଫେରନ୍ତୁ fp32.as_bits;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    ଅଚଳ uint32_t w = (uint32_t) h << 16;
    ଅଚଳ uint32_t sign = w & UINT32_C(0x80000000);
    ଅଚଳ uint32_t two_w = w + w;

    ଅଚଳ uint32_t exp_offset = UINT32_C(0xE0) << 23;
#ଯଦି (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    ଅଚଳ ଦଶମିକ exp_scale = 0x1.0p-112f;
#ଅନ୍ୟଥା
    ଅଚଳ ଦଶମିକ exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    ଅଚଳ ଦଶମିକ normalized_ମୂଲ୍ୟ = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    ଅଚଳ uint32_t magic_mask = UINT32_C(126) << 23;
    ଅଚଳ ଦଶମିକ magic_bias = 0.5f;
    ଅଚଳ ଦଶମିକ denormalized_ମୂଲ୍ୟ = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    ଅଚଳ uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    ଅଚଳ uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_ମୂଲ୍ୟ) : fp32_to_bits(normalized_ମୂଲ୍ୟ));
    ଫେରନ୍ତୁ fp32_from_bits(result);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ggml_fp16_t ggml_compute_fp32_to_fp16(ଦଶମିକ f) {
#ଯଦି (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    ଅଚଳ ଦଶମିକ scale_to_inf = 0x1.0p+112f;
    ଅଚଳ ଦଶମିକ scale_to_zero = 0x1.0p-110f;
#ଅନ୍ୟଥା
    ଅଚଳ ଦଶମିକ scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    ଅଚଳ ଦଶମିକ scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    ଦଶମିକ base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    ଅଚଳ uint32_t w = fp32_to_bits(f);
    ଅଚଳ uint32_t shl1_w = w + w;
    ଅଚଳ uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    ଯଦି (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    ଅଚଳ uint32_t bits = fp32_to_bits(base);
    ଅଚଳ uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    ଅଚଳ uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    ଅଚଳ uint32_t nonsign = exp_bits + mantissa_bits;
    ଫେରନ୍ତୁ (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ ggml_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;  // Stores the raw bit representation of the float

    // Handle special case for minimum exponent (denormalized float)
    ଯଦି (x == 0) {
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
    ଅନ୍ୟଥା {
        // Construct normalized float by shifting exponent into position:
        // - Exponent field: 8 bits (positions 30-23)
        // - Mantissa: 0 (implicit leading 1)
        // Value = 2^(x - 127)
        bits = (uint32_t) x << 23;
    }

    ଦଶମିକ result;  // Final float value
                   // Safely reinterpret bit pattern as float without type-punning issues
    memcpy(&result, &bits, ଆକାର(ଦଶମିକ));
    ଫେରନ୍ତୁ result;
}

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ ggml_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;

    // For x < 2: use precomputed denormal patterns
    ଯଦି (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        bits = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    ଅନ୍ୟଥା {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        bits = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    ଦଶମିକ result;
    memcpy(&result, &bits, ଆକାର(ଦଶମିକ));
    ଫେରନ୍ତୁ result;
}

#define GGML_E8M0_TO_FP32(x) ggml_e8m0_to_fp32(x)
#define GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ ggml_ue4m3_to_fp32(uint8_t x) {
    ଯଦି (x == 0 || x == 0x7F) {
        ଫେରନ୍ତୁ 0.0f;
    }
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା   exp = (x >> 3) & 0xF;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା   man = x & 0x7;
    ଦଶମିକ raw;
    ଯଦି (exp == 0) {
        raw = ldexpf((ଦଶମିକ) man, -9);
    } ଅନ୍ୟଥା {
        raw = ldexpf(1.0f + (ଦଶମିକ) man / 8.0f, exp - 7);
    }
    ଫେରନ୍ତୁ raw * 0.5f;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint8_t ggml_fp32_to_ue4m3(ଦଶମିକ x) {
    ଯଦି (!(x > 0.0f)) {
        ଫେରନ୍ତୁ 0;
    }
    ଯଦି (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t bits;
    memcpy(&bits, &x, 4);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା fp32_exp  = ((bits >> 23) & 0xFF) - 127;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା fp32_man  = (bits >> 20) & 0x7;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ue4m3_exp = fp32_exp + 7;
    ଯଦି (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା man = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (x * 512.0f + 0.5f);
        ଯଦି (man > 7) {
            man = 7;
        }
        ଯଦି (man < 1) {
            ଫେରନ୍ତୁ 0;
        }
        ଫେରନ୍ତୁ (uint8_t) man;
    }
    ଯଦି (ue4m3_exp >= 15) {
        ଫେରନ୍ତୁ 0x7E;
    }
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା round_bit = (bits >> 19) & 1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ue4m3_man = fp32_man + round_bit;
    ଯଦି (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        ଯଦି (ue4m3_exp >= 15) {
            ଫେରନ୍ତୁ 0x7E;
        }
    }
    ଫେରନ୍ତୁ (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
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
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    ସଂଘ {
        ଦଶମିକ f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    ଫେରନ୍ତୁ u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ggml_bf16_t ggml_compute_fp32_to_bf16(ଦଶମିକ s) {
    ggml_bf16_t h;
    ସଂଘ {
        ଦଶମିକ f;
        uint32_t i;
    } u;
    u.f = s;
    ଯଦି ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.bits = (u.i >> 16) | 64; /* force to quiet */
        ଫେରନ୍ତୁ h;
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    ଫେରନ୍ତୁ h;
}

#define GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)
#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)

ସ୍ଥିର ଧାଡ଼ିଭିତରେ int32_t ggml_ଗଣ୍ଠି_get_use_ଗଣନା(ଅଚଳ ସଂରଚନା ggml_cgraph * cgraph, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) {
    ଅଚଳ ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[ଗଣ୍ଠି_idx];

    size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, ଗଣ୍ଠି);
    ଯଦି (!ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        ଫେରନ୍ତୁ 0;
    }
    ଫେରନ୍ତୁ cgraph->use_counts[hash_pos];
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool ggml_ଗଣ୍ଠି_has_n_uses(ଅଚଳ ସଂରଚନା ggml_cgraph * cgraph, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, int32_t n_uses) {
    ଅଚଳ ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[ଗଣ୍ଠି_idx];

    // check the use count against how many we're replacing
    ଯଦି (ggml_ଗଣ୍ଠି_get_use_ଗଣନା(cgraph, ଗଣ୍ଠି_idx) != n_uses) {
        ଫେରନ୍ତୁ false;
    }

    // if node is a view, some other node might be using the intermediate result
    // via the view source.
    ଯଦି (ଗଣ୍ଠି->view_src) {
        ଫେରନ୍ତୁ false;
    }

    // If the user requested output for the node, can't fuse
    ଯଦି (ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_OUTPUT) {
        ଫେରନ୍ତୁ false;
    }

    ଫେରନ୍ତୁ true;
}

// Returns true if nodes with indices { node_idxs } are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool ggml_can_fuse_ext(ଅଚଳ ସଂରଚନା ggml_cgraph * cgraph, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା * ଗଣ୍ଠି_idxs, ଅଚଳ ଗଣନା ggml_op * ops, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_ops) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ଅଙ୍କ_ops; ++i) {
        ଯଦି (ଗଣ୍ଠି_idxs[i] >= cgraph->n_nodes) {
            ଫେରନ୍ତୁ false;
        }

        ସଂରଚନା ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[ଗଣ୍ଠି_idxs[i]];
        ଯଦି (ଗଣ୍ଠି->op != ops[i]) {
            ଫେରନ୍ତୁ false;
        }
        ଯଦି ((ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ଫେରନ୍ତୁ false;
        }
        ଯଦି (i < ଅଙ୍କ_ops - 1 && !ggml_ଗଣ୍ଠି_has_n_uses(cgraph, ଗଣ୍ଠି_idxs[i], 1)) {
            ଫେରନ୍ତୁ false;
        }
        ଯଦି (i > 0) {
            ସଂରଚନା ggml_tensor * prev = cgraph->nodes[ଗଣ୍ଠି_idxs[i - 1]];
            ଯଦି (ଗଣ୍ଠି->src[0] != prev && ଗଣ୍ଠି->src[1] != prev) {
                ଫେରନ୍ତୁ false;
            }
            ଯଦି (!ggml_are_same_shape(ଗଣ୍ଠି, prev)) {
                ଫେରନ୍ତୁ false;
            }
        }
    }
    ଫେରନ୍ତୁ true;
}

// same as above, for sequential indices starting at node_idx
ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool ggml_can_fuse(ଅଚଳ ସଂରଚନା ggml_cgraph * cgraph, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, ଅଚଳ ଗଣନା ggml_op * ops, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_ops) {
    ପ୍ରତିଜ୍ଞା(ଅଙ୍କ_ops < 32);

    ଯଦି (ଗଣ୍ଠି_idx + ଅଙ୍କ_ops > cgraph->n_nodes) {
        ଫେରନ୍ତୁ false;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା idxs[32];
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ଅଙ୍କ_ops; ++i) {
        idxs[i] = ଗଣ୍ଠି_idx + i;
    }

    ଫେରନ୍ତୁ ggml_can_fuse_ext(cgraph, idxs, ops, ଅଙ୍କ_ops);
}

GGML_API bool ggml_can_fuse_subgraph_ext(ଅଚଳ ସଂରଚନା ggml_cgraph * cgraph,
                                         ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା *                ଗଣ୍ଠି_idxs,
                                         ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        ଗଣନା_2,
                                         ଅଚଳ ଗଣନା ggml_op *       ops,
                                         ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା *                outputs,
                                         ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        ଅଙ୍କ_outputs);

// Returns true if the subgraph formed by {node_idxs} can be fused
// checks whethers all nodes which are not part of outputs can be elided
// by checking if their num_uses are confined to the subgraph
ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool ggml_can_fuse_subgraph(ଅଚଳ ସଂରଚନା ggml_cgraph * cgraph,
                                          ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        ଗଣ୍ଠି_idx,
                                          ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        ଗଣନା_2,
                                          ଅଚଳ ଗଣନା ggml_op *       ops,
                                          ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା *                outputs,
                                          ପୂର୍ଣ୍ଣସଂଖ୍ୟା                        ଅଙ୍କ_outputs) {
    GGML_ASSERT(ଗଣନା_2 < 32);
    ଯଦି (ଗଣ୍ଠି_idx + ଗଣନା_2 > cgraph->n_nodes) {
        ଫେରନ୍ତୁ false;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା idxs[32];

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ଗଣନା_2; ++i) {
        idxs[i] = ଗଣ୍ଠି_idx + i;
    }

    ଫେରନ୍ତୁ ggml_can_fuse_subgraph_ext(cgraph, idxs, ଗଣନା_2, ops, outputs, ଅଙ୍କ_outputs);
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#ଅନ୍ତର୍ଭାଵ <array>
#ଅନ୍ତର୍ଭାଵ <initializer_list>
#ଅନ୍ତର୍ଭାଵ <vector>

// nicer C++ syntax for ggml_can_fuse
ଧାଡ଼ିଭିତରେ bool ggml_can_fuse(ଅଚଳ ସଂରଚନା ggml_cgraph * cgraph, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, std::initializer_list<ଗଣନା ggml_op> ops) {
    ଫେରନ୍ତୁ ggml_can_fuse(cgraph, ଗଣ୍ଠି_idx, ops.begin(), (ପୂର୍ଣ୍ଣସଂଖ୍ୟା)ops.size());
}

ଧାଡ଼ିଭିତରେ bool ggml_can_fuse_subgraph(ଅଚଳ ସଂରଚନା ggml_cgraph *          cgraph,
                                   ପୂର୍ଣ୍ଣସଂଖ୍ୟା                                 start_idx,
                                   std::initializer_list<ଗଣନା ggml_op> ops,
                                   std::initializer_list<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>          outputs = {}) {
    ଫେରନ୍ତୁ ggml_can_fuse_subgraph(cgraph, start_idx, ops.size(), ops.begin(), outputs.begin(), outputs.size());
}

// Return true if the edges in the graph match expectations.
ଧାଡ଼ିଭିତରେ bool ggml_check_edges(ଅଚଳ ସଂରଚନା ggml_cgraph *                cgraph,
                             ପୂର୍ଣ୍ଣସଂଖ୍ୟା                                       start_idx,
                             std::initializer_list<std::array<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, 3>> edges) {
    ପାଇଁ (ଅଚଳ auto & edge : edges) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା dst_ଗଣ୍ଠି = edge[0];
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_idx  = edge[1];
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_ଗଣ୍ଠି = edge[2];
        ଯଦି (cgraph->nodes[start_idx + dst_ଗଣ୍ଠି]->src[src_idx] != cgraph->nodes[start_idx + src_ଗଣ୍ଠି]) {
            ଫେରନ୍ତୁ false;
        }
    }
    ଫେରନ୍ତୁ true;
}

// expose GGUF internals for test code
GGML_API size_t gguf_type_size(ଗଣନା gguf_type type);
GGML_API ଶୂନ୍ୟ gguf_write_to_buf(ଅଚଳ ସଂରଚନା gguf_context * ctx, std::vector<int8_t> & buf, bool only_meta);
#endif // __cplusplus
