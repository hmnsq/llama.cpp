#pragma once

// GGML internal header

#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "gguf.h"

#અન્તર્ભાવ <assert.h>
#અન્તર્ભાવ <math.h>
#અન્તર્ભાવ <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#અન્તર્ભાવ <stdbool.h>
#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <string.h>

#ifdef __ARM_FEATURE_SVE
#અન્તર્ભાવ <arm_sve.h>
#endif // __ARM_FEATURE_SVE

#જો defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#અન્તર્ભાવ <arm_neon.h>
#endif

#ifdef __cplusplus
બાહ્ય "C" {
#endif

શૂન્ય ggml_print_backtrace(શૂન્ય);

uint64_t ggml_graph_next_uid(શૂન્ય);

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
        #જો defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
            #define static_assert(cond, સંદેશ) _Static_assert(cond, સંદેશ)
        #નહીંતર
            #define static_assert(cond, સંદેશ) સંરચના global_scope_noop_trick
        #endif
    #endif
#endif

સ્થિર અંતઃરેખ પૂર્ણાંક ggml_up32(પૂર્ણાંક n) {
    પાછા (n + 31) & ~31;
}

//static inline int ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

સ્થિર અંતઃરેખ પૂર્ણાંક ggml_up(પૂર્ણાંક n, પૂર્ણાંક m) {
    // assert m is a power of 2
    GGML_ASSERT((m & (m - 1)) == 0);
    પાછા (n + m - 1) & ~(m - 1);
}

// TODO: move to ggml.h? (won't be able to inline)
સ્થિર bool ggml_are_same_layout(અચલ સંરચના ggml_tensor * a, અચલ સંરચના ggml_tensor * b) {
    જો (a->type != b->type) {
        પાછા false;
    }
    માટે (પૂર્ણાંક i = 0; i < GGML_MAX_DIMS; i++) {
        જો (a->ne[i] != b->ne[i]) {
            પાછા false;
        }
        જો (a->nb[i] != b->nb[i]) {
            પાછા false;
        }
    }
    પાછા true;
}

સ્થિર bool ggml_op_is_empty(ગણના ggml_op op) {
    પસંદગી (op) {
        સ્થિતિ GGML_OP_NONE:
        સ્થિતિ GGML_OP_RESHAPE:
        સ્થિતિ GGML_OP_TRANSPOSE:
        સ્થિતિ GGML_OP_VIEW:
        સ્થિતિ GGML_OP_PERMUTE:
            પાછા true;
        મૂળભૂત:
            પાછા false;
    }
}

સ્થિર અંતઃરેખ bool ggml_impl_is_view(અચલ સંરચના ggml_tensor * t) {
    પાછા t->view_src != ખાલી;
}

સ્થિર અંતઃરેખ દશાંશ ggml_compute_softplus_f32(દશાંશ આગત) {
    પાછા (આગત > 20.0f) ? આગત : logf(1 + expf(આગત));
}
//
// logging
//

GGML_ATTRIBUTE_FORMAT(2, 3)
GGML_API શૂન્ય ggml_log_internal        (ગણના ggml_log_level level, અચલ અક્ષર * format, ...);
GGML_API શૂન્ય ggml_log_callback_default(ગણના ggml_log_level level, અચલ અક્ષર * text, શૂન્ય * user_data);

#define GGML_LOG(...)       ggml_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define GGML_LOG_INFO(...)  ggml_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define GGML_LOG_WARN(...)  ggml_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define GGML_LOG_ERROR(...) ggml_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GGML_LOG_DEBUG(...) ggml_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define GGML_LOG_CONT(...)  ggml_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

#define GGML_DEBUG 0

#જો (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#નહીંતર
#define GGML_PRINT_DEBUG(...)
#endif

#જો (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) GGML_LOG_DEBUG(__VA_ARGS__)
#નહીંતર
#define GGML_PRINT_DEBUG_5(...)
#endif

#જો (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) GGML_LOG_DEBUG(__VA_ARGS__)
#નહીંતર
#define GGML_PRINT_DEBUG_10(...)
#endif

// tensor params

સ્થિર શૂન્ય ggml_set_op_params(સંરચના ggml_tensor * tensor, અચલ શૂન્ય * params, size_t params_size) {
    GGML_ASSERT(tensor != ખાલી); // silence -Warray-bounds warnings
    પ્રતિજ્ઞા(params_size <= GGML_MAX_OP_PARAMS);
    memcpy(tensor->op_params, params, params_size);
}

સ્થિર int32_t ggml_get_op_params_i32(અચલ સંરચના ggml_tensor * tensor, uint32_t i) {
    પ્રતિજ્ઞા(i < GGML_MAX_OP_PARAMS / કદ(int32_t));
    પાછા ((અચલ int32_t *)(tensor->op_params))[i];
}

સ્થિર દશાંશ ggml_get_op_params_f32(અચલ સંરચના ggml_tensor * tensor, uint32_t i) {
    પ્રતિજ્ઞા(i < GGML_MAX_OP_PARAMS / કદ(દશાંશ));
    પાછા ((અચલ દશાંશ *)(tensor->op_params))[i];
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
સ્થિર શૂન્ય ggml_set_op_params_i32(સંરચના ggml_tensor * tensor, uint32_t i, int32_t કિંમત) {
    પ્રતિજ્ઞા(i < GGML_MAX_OP_PARAMS / કદ(int32_t));
    ((int32_t *)(tensor->op_params))[i] = કિંમત;
}

સ્થિર શૂન્ય ggml_set_op_params_f32(સંરચના ggml_tensor * tensor, uint32_t i, દશાંશ કિંમત) {
    પ્રતિજ્ઞા(i < GGML_MAX_OP_PARAMS / કદ(દશાંશ));
    ((દશાંશ *)(tensor->op_params))[i] = કિંમત;
}

સંરચના ggml_map_custom1_op_params {
    ggml_custom1_op_t  fun;
    પૂર્ણાંક                n_tasks;
    શૂન્ય             * userdata;
};

સંરચના ggml_map_custom2_op_params {
    ggml_custom2_op_t   fun;
    પૂર્ણાંક                 n_tasks;
    શૂન્ય              * userdata;
};

સંરચના ggml_map_custom3_op_params {
    ggml_custom3_op_t fun;
    પૂર્ણાંક               n_tasks;
    શૂન્ય            * userdata;
};

સંરચના ggml_custom_op_params {
    ggml_custom_op_t fun;
    પૂર્ણાંક              n_tasks;
    શૂન્ય           * userdata;
};

// bitset

પ્રકારવ્યાખ્યા uint32_t ggml_bitset_t;

static_assert(કદ(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#define BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#define BITSET_MASK (કદ(ggml_bitset_t)*8 - 1)

સ્થિર size_t ggml_bitset_size(size_t n) {
    પાછા (n + BITSET_MASK) >> BITSET_SHR;
}

સ્થિર અંતઃરેખ bool ggml_bitset_get(અચલ ggml_bitset_t * bitset, size_t i) {
    પાછા !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
}

સ્થિર અંતઃરેખ શૂન્ય ggml_bitset_set(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_MASK));
}

સ્થિર અંતઃરેખ શૂન્ય ggml_bitset_clear(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_MASK));
}

// hash set

#define GGML_HASHSET_FULL ((size_t)-1)
#define GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)

સંરચના ggml_hash_set {
    size_t size;
    ggml_bitset_t * used;       // whether or not the keys are in use i.e. set
    સંરચના ggml_tensor ** ચાવીઓ; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};

સંરચના ggml_hash_set ggml_hash_set_new(size_t size);
શૂન્ય                 ggml_hash_set_free(સંરચના ggml_hash_set * hash_set);

// returns the minimum size for a hash set that can hold min_sz elements
size_t ggml_hash_size(size_t min_sz);

// remove all elements from the hash set
શૂન્ય ggml_hash_set_reset(સંરચના ggml_hash_set * hash_set);

// returns true if key is in the hash set
સ્થિર bool ggml_hash_contains(અચલ સંરચના ggml_hash_set * hash_set, સંરચના ggml_tensor * ચાવી);

// returns GGML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
સ્થિર size_t ggml_hash_find(અચલ સંરચના ggml_hash_set * hash_set, અચલ સંરચના ggml_tensor * ચાવી);

// returns GGML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
સ્થિર size_t ggml_hash_ઉમેરો(સંરચના ggml_hash_set * hash_set, સંરચના ggml_tensor * ચાવી);

// return index, asserts if table is full
સ્થિર size_t ggml_hash_find_or_ઉમેરો(સંરચના ggml_hash_set * hash_set, સંરચના ggml_tensor * ચાવી);

// hash function for ggml_tensor
સ્થિર અંતઃરેખ size_t ggml_hash(અચલ સંરચના ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    પાછા (size_t)(uintptr_t)p >> 4;
}

સ્થિર size_t ggml_hash_find(અચલ સંરચના ggml_hash_set * hash_set, અચલ સંરચના ggml_tensor * ચાવી) {
    size_t h = ggml_hash(ચાવી) % hash_set->size;

    // linear probing
    size_t i = h;
    જ્યારે (ggml_bitset_get(hash_set->used, i) && hash_set->ચાવીઓ[i] != ચાવી) {
        i = (i + 1) % hash_set->size;
        જો (i == h) {
            // visited all hash table entries -> not found
            પાછા GGML_HASHSET_FULL;
        }
    }
    પાછા i;
}

સ્થિર bool ggml_hash_contains(અચલ સંરચના ggml_hash_set * hash_set, સંરચના ggml_tensor * ચાવી) {
    size_t i = ggml_hash_find(hash_set, ચાવી);
    પાછા i != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, i);
}

સ્થિર size_t ggml_hash_ઉમેરો(સંરચના ggml_hash_set * hash_set, સંરચના ggml_tensor * ચાવી) {
    size_t h = ggml_hash(ચાવી) % hash_set->size;

    // linear probing
    size_t i = h;
    કરો {
        જો (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->ચાવીઓ[i] = ચાવી;
            પાછા i;
        }
        જો (hash_set->ચાવીઓ[i] == ચાવી) {
            પાછા GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_set->size;
    } જ્યારે (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

સ્થિર size_t ggml_hash_find_or_ઉમેરો(સંરચના ggml_hash_set * hash_set, સંરચના ggml_tensor * ચાવી) {
    size_t h = ggml_hash(ચાવી) % hash_set->size;

    // linear probing
    size_t i = h;
    કરો {
        જો (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->ચાવીઓ[i] = ચાવી;
            પાછા i;
        }
        જો (hash_set->ચાવીઓ[i] == ચાવી) {
            પાછા i;
        }
        i = (i + 1) % hash_set->size;
    } જ્યારે (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

// computation graph

ગણના ggml_cgraph_eval_order {
    GGML_CGRAPH_EVAL_ORDER_ડાબું_TO_જમણું = 0,
    GGML_CGRAPH_EVAL_ORDER_જમણું_TO_ડાબું,
    GGML_CGRAPH_EVAL_ORDER_ગણતરી
};

સંરચના ggml_cgraph {
    પૂર્ણાંક size;    // maximum number of nodes/leafs/grads/grad_accs
    પૂર્ણાંક n_nodes; // number of nodes currently in use
    પૂર્ણાંક n_leafs; // number of leafs currently in use

    સંરચના ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
    સંરચના ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
    સંરચના ggml_tensor ** grad_accs; // accumulators for node gradients
    સંરચના ggml_tensor ** leafs;     // tensors with constant data
    int32_t             * use_counts;// number of uses of each tensor, indexed by hash table slot

    સંરચના ggml_hash_set visited_hash_set;

    ગણના ggml_cgraph_eval_order order;

    // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
    // a value of 0 means it is not set and should be ignored
    uint64_t uid;
};

// returns a slice of cgraph with nodes [i0, i1)
// the slice does not have leafs or gradients
// if you need the gradients, get them from the original graph
સંરચના ggml_cgraph ggml_graph_view(સંરચના ggml_cgraph * cgraph, પૂર્ણાંક i0, પૂર્ણાંક i1);

// ggml-alloc.c: true if the operation can reuse memory from its sources
GGML_API bool ggml_op_can_inplace(ગણના ggml_op op);


// Memory allocation

GGML_API શૂન્ય * ggml_aligned_malloc(size_t size);
GGML_API શૂન્ય ggml_aligned_free(શૂન્ય * ptr, size_t size);

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

સ્થિર અંતઃરેખ દશાંશ fp32_from_bits(uint32_t w) {
    સંઘ {
        uint32_t as_bits;
        દશાંશ as_કિંમત;
    } fp32;
    fp32.as_bits = w;
    પાછા fp32.as_કિંમત;
}

સ્થિર અંતઃરેખ uint32_t fp32_to_bits(દશાંશ f) {
    સંઘ {
        દશાંશ as_કિંમત;
        uint32_t as_bits;
    } fp32;
    fp32.as_કિંમત = f;
    પાછા fp32.as_bits;
}

સ્થિર અંતઃરેખ દશાંશ ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    અચલ uint32_t w = (uint32_t) h << 16;
    અચલ uint32_t sign = w & UINT32_C(0x80000000);
    અચલ uint32_t two_w = w + w;

    અચલ uint32_t exp_offset = UINT32_C(0xE0) << 23;
#જો (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    અચલ દશાંશ exp_scale = 0x1.0p-112f;
#નહીંતર
    અચલ દશાંશ exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    અચલ દશાંશ normalized_કિંમત = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    અચલ uint32_t magic_mask = UINT32_C(126) << 23;
    અચલ દશાંશ magic_bias = 0.5f;
    અચલ દશાંશ denormalized_કિંમત = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    અચલ uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    અચલ uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_કિંમત) : fp32_to_bits(normalized_કિંમત));
    પાછા fp32_from_bits(result);
}

સ્થિર અંતઃરેખ ggml_fp16_t ggml_compute_fp32_to_fp16(દશાંશ f) {
#જો (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    અચલ દશાંશ scale_to_inf = 0x1.0p+112f;
    અચલ દશાંશ scale_to_zero = 0x1.0p-110f;
#નહીંતર
    અચલ દશાંશ scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    અચલ દશાંશ scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    દશાંશ base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    અચલ uint32_t w = fp32_to_bits(f);
    અચલ uint32_t shl1_w = w + w;
    અચલ uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    જો (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    અચલ uint32_t bits = fp32_to_bits(base);
    અચલ uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    અચલ uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    અચલ uint32_t nonsign = exp_bits + mantissa_bits;
    પાછા (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

સ્થિર અંતઃરેખ દશાંશ ggml_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;  // Stores the raw bit representation of the float

    // Handle special case for minimum exponent (denormalized float)
    જો (x == 0) {
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
    નહીંતર {
        // Construct normalized float by shifting exponent into position:
        // - Exponent field: 8 bits (positions 30-23)
        // - Mantissa: 0 (implicit leading 1)
        // Value = 2^(x - 127)
        bits = (uint32_t) x << 23;
    }

    દશાંશ result;  // Final float value
                   // Safely reinterpret bit pattern as float without type-punning issues
    memcpy(&result, &bits, કદ(દશાંશ));
    પાછા result;
}

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
સ્થિર અંતઃરેખ દશાંશ ggml_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;

    // For x < 2: use precomputed denormal patterns
    જો (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        bits = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    નહીંતર {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        bits = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    દશાંશ result;
    memcpy(&result, &bits, કદ(દશાંશ));
    પાછા result;
}

#define GGML_E8M0_TO_FP32(x) ggml_e8m0_to_fp32(x)
#define GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
સ્થિર અંતઃરેખ દશાંશ ggml_ue4m3_to_fp32(uint8_t x) {
    જો (x == 0 || x == 0x7F) {
        પાછા 0.0f;
    }
    પૂર્ણાંક   exp = (x >> 3) & 0xF;
    પૂર્ણાંક   man = x & 0x7;
    દશાંશ raw;
    જો (exp == 0) {
        raw = ldexpf((દશાંશ) man, -9);
    } નહીંતર {
        raw = ldexpf(1.0f + (દશાંશ) man / 8.0f, exp - 7);
    }
    પાછા raw * 0.5f;
}

સ્થિર અંતઃરેખ uint8_t ggml_fp32_to_ue4m3(દશાંશ x) {
    જો (!(x > 0.0f)) {
        પાછા 0;
    }
    જો (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t bits;
    memcpy(&bits, &x, 4);
    પૂર્ણાંક fp32_exp  = ((bits >> 23) & 0xFF) - 127;
    પૂર્ણાંક fp32_man  = (bits >> 20) & 0x7;
    પૂર્ણાંક ue4m3_exp = fp32_exp + 7;
    જો (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        પૂર્ણાંક man = (પૂર્ણાંક) (x * 512.0f + 0.5f);
        જો (man > 7) {
            man = 7;
        }
        જો (man < 1) {
            પાછા 0;
        }
        પાછા (uint8_t) man;
    }
    જો (ue4m3_exp >= 15) {
        પાછા 0x7E;
    }
    પૂર્ણાંક round_bit = (bits >> 19) & 1;
    પૂર્ણાંક ue4m3_man = fp32_man + round_bit;
    જો (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        જો (ue4m3_exp >= 15) {
            પાછા 0x7E;
        }
    }
    પાછા (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
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
સ્થિર અંતઃરેખ દશાંશ ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    સંઘ {
        દશાંશ f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    પાછા u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
સ્થિર અંતઃરેખ ggml_bf16_t ggml_compute_fp32_to_bf16(દશાંશ s) {
    ggml_bf16_t h;
    સંઘ {
        દશાંશ f;
        uint32_t i;
    } u;
    u.f = s;
    જો ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.bits = (u.i >> 16) | 64; /* force to quiet */
        પાછા h;
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    પાછા h;
}

#define GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)
#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)

સ્થિર અંતઃરેખ int32_t ggml_ગાંઠ_get_use_ગણતરી(અચલ સંરચના ggml_cgraph * cgraph, પૂર્ણાંક ગાંઠ_idx) {
    અચલ સંરચના ggml_tensor * ગાંઠ = cgraph->nodes[ગાંઠ_idx];

    size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, ગાંઠ);
    જો (!ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        પાછા 0;
    }
    પાછા cgraph->use_counts[hash_pos];
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
સ્થિર અંતઃરેખ bool ggml_ગાંઠ_has_n_uses(અચલ સંરચના ggml_cgraph * cgraph, પૂર્ણાંક ગાંઠ_idx, int32_t n_uses) {
    અચલ સંરચના ggml_tensor * ગાંઠ = cgraph->nodes[ગાંઠ_idx];

    // check the use count against how many we're replacing
    જો (ggml_ગાંઠ_get_use_ગણતરી(cgraph, ગાંઠ_idx) != n_uses) {
        પાછા false;
    }

    // if node is a view, some other node might be using the intermediate result
    // via the view source.
    જો (ગાંઠ->view_src) {
        પાછા false;
    }

    // If the user requested output for the node, can't fuse
    જો (ગાંઠ->flags & GGML_TENSOR_FLAG_OUTPUT) {
        પાછા false;
    }

    પાછા true;
}

// Returns true if nodes with indices { node_idxs } are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
સ્થિર અંતઃરેખ bool ggml_can_fuse_ext(અચલ સંરચના ggml_cgraph * cgraph, અચલ પૂર્ણાંક * ગાંઠ_idxs, અચલ ગણના ggml_op * ops, પૂર્ણાંક અંક_ops) {
    માટે (પૂર્ણાંક i = 0; i < અંક_ops; ++i) {
        જો (ગાંઠ_idxs[i] >= cgraph->n_nodes) {
            પાછા false;
        }

        સંરચના ggml_tensor * ગાંઠ = cgraph->nodes[ગાંઠ_idxs[i]];
        જો (ગાંઠ->op != ops[i]) {
            પાછા false;
        }
        જો ((ગાંઠ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            પાછા false;
        }
        જો (i < અંક_ops - 1 && !ggml_ગાંઠ_has_n_uses(cgraph, ગાંઠ_idxs[i], 1)) {
            પાછા false;
        }
        જો (i > 0) {
            સંરચના ggml_tensor * prev = cgraph->nodes[ગાંઠ_idxs[i - 1]];
            જો (ગાંઠ->src[0] != prev && ગાંઠ->src[1] != prev) {
                પાછા false;
            }
            જો (!ggml_are_same_shape(ગાંઠ, prev)) {
                પાછા false;
            }
        }
    }
    પાછા true;
}

// same as above, for sequential indices starting at node_idx
સ્થિર અંતઃરેખ bool ggml_can_fuse(અચલ સંરચના ggml_cgraph * cgraph, પૂર્ણાંક ગાંઠ_idx, અચલ ગણના ggml_op * ops, પૂર્ણાંક અંક_ops) {
    પ્રતિજ્ઞા(અંક_ops < 32);

    જો (ગાંઠ_idx + અંક_ops > cgraph->n_nodes) {
        પાછા false;
    }

    પૂર્ણાંક idxs[32];
    માટે (પૂર્ણાંક i = 0; i < અંક_ops; ++i) {
        idxs[i] = ગાંઠ_idx + i;
    }

    પાછા ggml_can_fuse_ext(cgraph, idxs, ops, અંક_ops);
}

GGML_API bool ggml_can_fuse_subgraph_ext(અચલ સંરચના ggml_cgraph * cgraph,
                                         અચલ પૂર્ણાંક *                ગાંઠ_idxs,
                                         પૂર્ણાંક                        ગણતરી,
                                         અચલ ગણના ggml_op *       ops,
                                         અચલ પૂર્ણાંક *                outputs,
                                         પૂર્ણાંક                        અંક_outputs);

// Returns true if the subgraph formed by {node_idxs} can be fused
// checks whethers all nodes which are not part of outputs can be elided
// by checking if their num_uses are confined to the subgraph
સ્થિર અંતઃરેખ bool ggml_can_fuse_subgraph(અચલ સંરચના ggml_cgraph * cgraph,
                                          પૂર્ણાંક                        ગાંઠ_idx,
                                          પૂર્ણાંક                        ગણતરી,
                                          અચલ ગણના ggml_op *       ops,
                                          અચલ પૂર્ણાંક *                outputs,
                                          પૂર્ણાંક                        અંક_outputs) {
    GGML_ASSERT(ગણતરી < 32);
    જો (ગાંઠ_idx + ગણતરી > cgraph->n_nodes) {
        પાછા false;
    }

    પૂર્ણાંક idxs[32];

    માટે (પૂર્ણાંક i = 0; i < ગણતરી; ++i) {
        idxs[i] = ગાંઠ_idx + i;
    }

    પાછા ggml_can_fuse_subgraph_ext(cgraph, idxs, ગણતરી, ops, outputs, અંક_outputs);
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#અન્તર્ભાવ <array>
#અન્તર્ભાવ <initializer_list>
#અન્તર્ભાવ <vector>

// nicer C++ syntax for ggml_can_fuse
અંતઃરેખ bool ggml_can_fuse(અચલ સંરચના ggml_cgraph * cgraph, પૂર્ણાંક ગાંઠ_idx, std::initializer_list<ગણના ggml_op> ops) {
    પાછા ggml_can_fuse(cgraph, ગાંઠ_idx, ops.begin(), (પૂર્ણાંક)ops.size());
}

અંતઃરેખ bool ggml_can_fuse_subgraph(અચલ સંરચના ggml_cgraph *          cgraph,
                                   પૂર્ણાંક                                 start_idx,
                                   std::initializer_list<ગણના ggml_op> ops,
                                   std::initializer_list<પૂર્ણાંક>          outputs = {}) {
    પાછા ggml_can_fuse_subgraph(cgraph, start_idx, ops.size(), ops.begin(), outputs.begin(), outputs.size());
}

// Return true if the edges in the graph match expectations.
અંતઃરેખ bool ggml_check_edges(અચલ સંરચના ggml_cgraph *                cgraph,
                             પૂર્ણાંક                                       start_idx,
                             std::initializer_list<std::array<પૂર્ણાંક, 3>> edges) {
    માટે (અચલ auto & edge : edges) {
        પૂર્ણાંક dst_ગાંઠ = edge[0];
        પૂર્ણાંક src_idx  = edge[1];
        પૂર્ણાંક src_ગાંઠ = edge[2];
        જો (cgraph->nodes[start_idx + dst_ગાંઠ]->src[src_idx] != cgraph->nodes[start_idx + src_ગાંઠ]) {
            પાછા false;
        }
    }
    પાછા true;
}

// expose GGUF internals for test code
GGML_API size_t gguf_type_size(ગણના gguf_type type);
GGML_API શૂન્ય gguf_write_to_buf(અચલ સંરચના gguf_context * ctx, std::vector<int8_t> & buf, bool only_meta);
#endif // __cplusplus
