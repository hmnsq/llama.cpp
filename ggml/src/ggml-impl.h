#pragma once

// GGML internal header

#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "gguf.h"

#அந்தர்பாவ <assert.h>
#அந்தர்பாவ <math.h>
#அந்தர்பாவ <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#அந்தர்பாவ <stdbool.h>
#அந்தர்பாவ <stdint.h>
#அந்தர்பாவ <string.h>

#ifdef __ARM_FEATURE_SVE
#அந்தர்பாவ <arm_sve.h>
#endif // __ARM_FEATURE_SVE

#எனில் defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#அந்தர்பாவ <arm_neon.h>
#endif

#ifdef __cplusplus
வெளி "C" {
#endif

வெறுமை ggml_print_backtrace(வெறுமை);

uint64_t ggml_graph_next_uid(வெறுமை);

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
        #எனில் defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
            #define static_assert(cond, செய்தி) _Static_assert(cond, செய்தி)
        #இல்லையெனில்
            #define static_assert(cond, செய்தி) கட்டமைப்பு global_scope_noop_trick
        #endif
    #endif
#endif

நிலையான உள்வரி முழு ggml_up32(முழு n) {
    திரும்பு (n + 31) & ~31;
}

//static inline int ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

நிலையான உள்வரி முழு ggml_up(முழு n, முழு m) {
    // assert m is a power of 2
    GGML_ASSERT((m & (m - 1)) == 0);
    திரும்பு (n + m - 1) & ~(m - 1);
}

// TODO: move to ggml.h? (won't be able to inline)
நிலையான bool ggml_are_same_layout(மாறிலி கட்டமைப்பு ggml_tensor * a, மாறிலி கட்டமைப்பு ggml_tensor * b) {
    எனில் (a->type != b->type) {
        திரும்பு false;
    }
    க்கு (முழு i = 0; i < GGML_MAX_DIMS; i++) {
        எனில் (a->ne[i] != b->ne[i]) {
            திரும்பு false;
        }
        எனில் (a->nb[i] != b->nb[i]) {
            திரும்பு false;
        }
    }
    திரும்பு true;
}

நிலையான bool ggml_op_is_empty(எண்ணிக்கை ggml_op op) {
    தேர்வு (op) {
        நிலை GGML_OP_NONE:
        நிலை GGML_OP_RESHAPE:
        நிலை GGML_OP_TRANSPOSE:
        நிலை GGML_OP_VIEW:
        நிலை GGML_OP_PERMUTE:
            திரும்பு true;
        இயல்பு:
            திரும்பு false;
    }
}

நிலையான உள்வரி bool ggml_impl_is_view(மாறிலி கட்டமைப்பு ggml_tensor * t) {
    திரும்பு t->view_src != வெற்று;
}

நிலையான உள்வரி தசம ggml_compute_softplus_f32(தசம உள்ளீடு) {
    திரும்பு (உள்ளீடு > 20.0f) ? உள்ளீடு : logf(1 + expf(உள்ளீடு));
}
//
// logging
//

GGML_ATTRIBUTE_FORMAT(2, 3)
GGML_API வெறுமை ggml_log_internal        (எண்ணிக்கை ggml_log_level level, மாறிலி எழுத்து * format, ...);
GGML_API வெறுமை ggml_log_callback_default(எண்ணிக்கை ggml_log_level level, மாறிலி எழுத்து * text, வெறுமை * user_data);

#define GGML_LOG(...)       ggml_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define GGML_LOG_INFO(...)  ggml_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define GGML_LOG_WARN(...)  ggml_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define GGML_LOG_ERROR(...) ggml_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GGML_LOG_DEBUG(...) ggml_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define GGML_LOG_CONT(...)  ggml_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

#define GGML_DEBUG 0

#எனில் (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#இல்லையெனில்
#define GGML_PRINT_DEBUG(...)
#endif

#எனில் (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) GGML_LOG_DEBUG(__VA_ARGS__)
#இல்லையெனில்
#define GGML_PRINT_DEBUG_5(...)
#endif

#எனில் (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) GGML_LOG_DEBUG(__VA_ARGS__)
#இல்லையெனில்
#define GGML_PRINT_DEBUG_10(...)
#endif

// tensor params

நிலையான வெறுமை ggml_set_op_params(கட்டமைப்பு ggml_tensor * tensor, மாறிலி வெறுமை * params, size_t params_size) {
    GGML_ASSERT(tensor != வெற்று); // silence -Warray-bounds warnings
    ப்ரதிஜ்ஞா(params_size <= GGML_MAX_OP_PARAMS);
    memcpy(tensor->op_params, params, params_size);
}

நிலையான int32_t ggml_get_op_params_i32(மாறிலி கட்டமைப்பு ggml_tensor * tensor, uint32_t i) {
    ப்ரதிஜ்ஞா(i < GGML_MAX_OP_PARAMS / அளவு(int32_t));
    திரும்பு ((மாறிலி int32_t *)(tensor->op_params))[i];
}

நிலையான தசம ggml_get_op_params_f32(மாறிலி கட்டமைப்பு ggml_tensor * tensor, uint32_t i) {
    ப்ரதிஜ்ஞா(i < GGML_MAX_OP_PARAMS / அளவு(தசம));
    திரும்பு ((மாறிலி தசம *)(tensor->op_params))[i];
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
நிலையான வெறுமை ggml_set_op_params_i32(கட்டமைப்பு ggml_tensor * tensor, uint32_t i, int32_t மதிப்பு) {
    ப்ரதிஜ்ஞா(i < GGML_MAX_OP_PARAMS / அளவு(int32_t));
    ((int32_t *)(tensor->op_params))[i] = மதிப்பு;
}

நிலையான வெறுமை ggml_set_op_params_f32(கட்டமைப்பு ggml_tensor * tensor, uint32_t i, தசம மதிப்பு) {
    ப்ரதிஜ்ஞா(i < GGML_MAX_OP_PARAMS / அளவு(தசம));
    ((தசம *)(tensor->op_params))[i] = மதிப்பு;
}

கட்டமைப்பு ggml_map_custom1_op_params {
    ggml_custom1_op_t  fun;
    முழு                n_tasks;
    வெறுமை             * userdata;
};

கட்டமைப்பு ggml_map_custom2_op_params {
    ggml_custom2_op_t   fun;
    முழு                 n_tasks;
    வெறுமை              * userdata;
};

கட்டமைப்பு ggml_map_custom3_op_params {
    ggml_custom3_op_t fun;
    முழு               n_tasks;
    வெறுமை            * userdata;
};

கட்டமைப்பு ggml_custom_op_params {
    ggml_custom_op_t fun;
    முழு              n_tasks;
    வெறுமை           * userdata;
};

// bitset

வகைவரை uint32_t ggml_bitset_t;

static_assert(அளவு(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#define BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#define BITSET_MASK (அளவு(ggml_bitset_t)*8 - 1)

நிலையான size_t ggml_bitset_size(size_t n) {
    திரும்பு (n + BITSET_MASK) >> BITSET_SHR;
}

நிலையான உள்வரி bool ggml_bitset_get(மாறிலி ggml_bitset_t * bitset, size_t i) {
    திரும்பு !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
}

நிலையான உள்வரி வெறுமை ggml_bitset_set(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_MASK));
}

நிலையான உள்வரி வெறுமை ggml_bitset_clear(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_MASK));
}

// hash set

#define GGML_HASHSET_FULL ((size_t)-1)
#define GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)

கட்டமைப்பு ggml_hash_set {
    size_t size;
    ggml_bitset_t * used;       // whether or not the keys are in use i.e. set
    கட்டமைப்பு ggml_tensor ** சாவிகள்; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};

கட்டமைப்பு ggml_hash_set ggml_hash_set_new(size_t size);
வெறுமை                 ggml_hash_set_free(கட்டமைப்பு ggml_hash_set * hash_set);

// returns the minimum size for a hash set that can hold min_sz elements
size_t ggml_hash_size(size_t min_sz);

// remove all elements from the hash set
வெறுமை ggml_hash_set_reset(கட்டமைப்பு ggml_hash_set * hash_set);

// returns true if key is in the hash set
நிலையான bool ggml_hash_contains(மாறிலி கட்டமைப்பு ggml_hash_set * hash_set, கட்டமைப்பு ggml_tensor * சாவி);

// returns GGML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
நிலையான size_t ggml_hash_find(மாறிலி கட்டமைப்பு ggml_hash_set * hash_set, மாறிலி கட்டமைப்பு ggml_tensor * சாவி);

// returns GGML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
நிலையான size_t ggml_hash_செருகு(கட்டமைப்பு ggml_hash_set * hash_set, கட்டமைப்பு ggml_tensor * சாவி);

// return index, asserts if table is full
நிலையான size_t ggml_hash_find_or_செருகு(கட்டமைப்பு ggml_hash_set * hash_set, கட்டமைப்பு ggml_tensor * சாவி);

// hash function for ggml_tensor
நிலையான உள்வரி size_t ggml_hash(மாறிலி கட்டமைப்பு ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    திரும்பு (size_t)(uintptr_t)p >> 4;
}

நிலையான size_t ggml_hash_find(மாறிலி கட்டமைப்பு ggml_hash_set * hash_set, மாறிலி கட்டமைப்பு ggml_tensor * சாவி) {
    size_t h = ggml_hash(சாவி) % hash_set->size;

    // linear probing
    size_t i = h;
    போது (ggml_bitset_get(hash_set->used, i) && hash_set->சாவிகள்[i] != சாவி) {
        i = (i + 1) % hash_set->size;
        எனில் (i == h) {
            // visited all hash table entries -> not found
            திரும்பு GGML_HASHSET_FULL;
        }
    }
    திரும்பு i;
}

நிலையான bool ggml_hash_contains(மாறிலி கட்டமைப்பு ggml_hash_set * hash_set, கட்டமைப்பு ggml_tensor * சாவி) {
    size_t i = ggml_hash_find(hash_set, சாவி);
    திரும்பு i != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, i);
}

நிலையான size_t ggml_hash_செருகு(கட்டமைப்பு ggml_hash_set * hash_set, கட்டமைப்பு ggml_tensor * சாவி) {
    size_t h = ggml_hash(சாவி) % hash_set->size;

    // linear probing
    size_t i = h;
    செய் {
        எனில் (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->சாவிகள்[i] = சாவி;
            திரும்பு i;
        }
        எனில் (hash_set->சாவிகள்[i] == சாவி) {
            திரும்பு GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_set->size;
    } போது (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

நிலையான size_t ggml_hash_find_or_செருகு(கட்டமைப்பு ggml_hash_set * hash_set, கட்டமைப்பு ggml_tensor * சாவி) {
    size_t h = ggml_hash(சாவி) % hash_set->size;

    // linear probing
    size_t i = h;
    செய் {
        எனில் (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->சாவிகள்[i] = சாவி;
            திரும்பு i;
        }
        எனில் (hash_set->சாவிகள்[i] == சாவி) {
            திரும்பு i;
        }
        i = (i + 1) % hash_set->size;
    } போது (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

// computation graph

எண்ணிக்கை ggml_cgraph_eval_order {
    GGML_CGRAPH_EVAL_ORDER_இடது_TO_வலது = 0,
    GGML_CGRAPH_EVAL_ORDER_வலது_TO_இடது,
    GGML_CGRAPH_EVAL_ORDER_மொத்தம்
};

கட்டமைப்பு ggml_cgraph {
    முழு size;    // maximum number of nodes/leafs/grads/grad_accs
    முழு n_nodes; // number of nodes currently in use
    முழு n_leafs; // number of leafs currently in use

    கட்டமைப்பு ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
    கட்டமைப்பு ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
    கட்டமைப்பு ggml_tensor ** grad_accs; // accumulators for node gradients
    கட்டமைப்பு ggml_tensor ** leafs;     // tensors with constant data
    int32_t             * use_counts;// number of uses of each tensor, indexed by hash table slot

    கட்டமைப்பு ggml_hash_set visited_hash_set;

    எண்ணிக்கை ggml_cgraph_eval_order order;

    // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
    // a value of 0 means it is not set and should be ignored
    uint64_t uid;
};

// returns a slice of cgraph with nodes [i0, i1)
// the slice does not have leafs or gradients
// if you need the gradients, get them from the original graph
கட்டமைப்பு ggml_cgraph ggml_graph_view(கட்டமைப்பு ggml_cgraph * cgraph, முழு i0, முழு i1);

// ggml-alloc.c: true if the operation can reuse memory from its sources
GGML_API bool ggml_op_can_inplace(எண்ணிக்கை ggml_op op);


// Memory allocation

GGML_API வெறுமை * ggml_aligned_malloc(size_t size);
GGML_API வெறுமை ggml_aligned_free(வெறுமை * ptr, size_t size);

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

நிலையான உள்வரி தசம fp32_from_bits(uint32_t w) {
    ஒன்றிப்பு {
        uint32_t as_bits;
        தசம as_மதிப்பு;
    } fp32;
    fp32.as_bits = w;
    திரும்பு fp32.as_மதிப்பு;
}

நிலையான உள்வரி uint32_t fp32_to_bits(தசம f) {
    ஒன்றிப்பு {
        தசம as_மதிப்பு;
        uint32_t as_bits;
    } fp32;
    fp32.as_மதிப்பு = f;
    திரும்பு fp32.as_bits;
}

நிலையான உள்வரி தசம ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    மாறிலி uint32_t w = (uint32_t) h << 16;
    மாறிலி uint32_t sign = w & UINT32_C(0x80000000);
    மாறிலி uint32_t two_w = w + w;

    மாறிலி uint32_t exp_offset = UINT32_C(0xE0) << 23;
#எனில் (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    மாறிலி தசம exp_scale = 0x1.0p-112f;
#இல்லையெனில்
    மாறிலி தசம exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    மாறிலி தசம normalized_மதிப்பு = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    மாறிலி uint32_t magic_mask = UINT32_C(126) << 23;
    மாறிலி தசம magic_bias = 0.5f;
    மாறிலி தசம denormalized_மதிப்பு = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    மாறிலி uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    மாறிலி uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_மதிப்பு) : fp32_to_bits(normalized_மதிப்பு));
    திரும்பு fp32_from_bits(result);
}

நிலையான உள்வரி ggml_fp16_t ggml_compute_fp32_to_fp16(தசம f) {
#எனில் (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    மாறிலி தசம scale_to_inf = 0x1.0p+112f;
    மாறிலி தசம scale_to_zero = 0x1.0p-110f;
#இல்லையெனில்
    மாறிலி தசம scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    மாறிலி தசம scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    தசம base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    மாறிலி uint32_t w = fp32_to_bits(f);
    மாறிலி uint32_t shl1_w = w + w;
    மாறிலி uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    எனில் (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    மாறிலி uint32_t bits = fp32_to_bits(base);
    மாறிலி uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    மாறிலி uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    மாறிலி uint32_t nonsign = exp_bits + mantissa_bits;
    திரும்பு (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

நிலையான உள்வரி தசம ggml_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;  // Stores the raw bit representation of the float

    // Handle special case for minimum exponent (denormalized float)
    எனில் (x == 0) {
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
    இல்லையெனில் {
        // Construct normalized float by shifting exponent into position:
        // - Exponent field: 8 bits (positions 30-23)
        // - Mantissa: 0 (implicit leading 1)
        // Value = 2^(x - 127)
        bits = (uint32_t) x << 23;
    }

    தசம result;  // Final float value
                   // Safely reinterpret bit pattern as float without type-punning issues
    memcpy(&result, &bits, அளவு(தசம));
    திரும்பு result;
}

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
நிலையான உள்வரி தசம ggml_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;

    // For x < 2: use precomputed denormal patterns
    எனில் (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        bits = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    இல்லையெனில் {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        bits = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    தசம result;
    memcpy(&result, &bits, அளவு(தசம));
    திரும்பு result;
}

#define GGML_E8M0_TO_FP32(x) ggml_e8m0_to_fp32(x)
#define GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
நிலையான உள்வரி தசம ggml_ue4m3_to_fp32(uint8_t x) {
    எனில் (x == 0 || x == 0x7F) {
        திரும்பு 0.0f;
    }
    முழு   exp = (x >> 3) & 0xF;
    முழு   man = x & 0x7;
    தசம raw;
    எனில் (exp == 0) {
        raw = ldexpf((தசம) man, -9);
    } இல்லையெனில் {
        raw = ldexpf(1.0f + (தசம) man / 8.0f, exp - 7);
    }
    திரும்பு raw * 0.5f;
}

நிலையான உள்வரி uint8_t ggml_fp32_to_ue4m3(தசம x) {
    எனில் (!(x > 0.0f)) {
        திரும்பு 0;
    }
    எனில் (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t bits;
    memcpy(&bits, &x, 4);
    முழு fp32_exp  = ((bits >> 23) & 0xFF) - 127;
    முழு fp32_man  = (bits >> 20) & 0x7;
    முழு ue4m3_exp = fp32_exp + 7;
    எனில் (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        முழு man = (முழு) (x * 512.0f + 0.5f);
        எனில் (man > 7) {
            man = 7;
        }
        எனில் (man < 1) {
            திரும்பு 0;
        }
        திரும்பு (uint8_t) man;
    }
    எனில் (ue4m3_exp >= 15) {
        திரும்பு 0x7E;
    }
    முழு round_bit = (bits >> 19) & 1;
    முழு ue4m3_man = fp32_man + round_bit;
    எனில் (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        எனில் (ue4m3_exp >= 15) {
            திரும்பு 0x7E;
        }
    }
    திரும்பு (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
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
நிலையான உள்வரி தசம ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    ஒன்றிப்பு {
        தசம f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    திரும்பு u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
நிலையான உள்வரி ggml_bf16_t ggml_compute_fp32_to_bf16(தசம s) {
    ggml_bf16_t h;
    ஒன்றிப்பு {
        தசம f;
        uint32_t i;
    } u;
    u.f = s;
    எனில் ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.bits = (u.i >> 16) | 64; /* force to quiet */
        திரும்பு h;
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    திரும்பு h;
}

#define GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)
#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)

நிலையான உள்வரி int32_t ggml_முனை_get_use_மொத்தம்(மாறிலி கட்டமைப்பு ggml_cgraph * cgraph, முழு முனை_idx) {
    மாறிலி கட்டமைப்பு ggml_tensor * முனை = cgraph->nodes[முனை_idx];

    size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, முனை);
    எனில் (!ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        திரும்பு 0;
    }
    திரும்பு cgraph->use_counts[hash_pos];
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
நிலையான உள்வரி bool ggml_முனை_has_n_uses(மாறிலி கட்டமைப்பு ggml_cgraph * cgraph, முழு முனை_idx, int32_t n_uses) {
    மாறிலி கட்டமைப்பு ggml_tensor * முனை = cgraph->nodes[முனை_idx];

    // check the use count against how many we're replacing
    எனில் (ggml_முனை_get_use_மொத்தம்(cgraph, முனை_idx) != n_uses) {
        திரும்பு false;
    }

    // if node is a view, some other node might be using the intermediate result
    // via the view source.
    எனில் (முனை->view_src) {
        திரும்பு false;
    }

    // If the user requested output for the node, can't fuse
    எனில் (முனை->flags & GGML_TENSOR_FLAG_OUTPUT) {
        திரும்பு false;
    }

    திரும்பு true;
}

// Returns true if nodes with indices { node_idxs } are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
நிலையான உள்வரி bool ggml_can_fuse_ext(மாறிலி கட்டமைப்பு ggml_cgraph * cgraph, மாறிலி முழு * முனை_idxs, மாறிலி எண்ணிக்கை ggml_op * ops, முழு இலக்கம்_ops) {
    க்கு (முழு i = 0; i < இலக்கம்_ops; ++i) {
        எனில் (முனை_idxs[i] >= cgraph->n_nodes) {
            திரும்பு false;
        }

        கட்டமைப்பு ggml_tensor * முனை = cgraph->nodes[முனை_idxs[i]];
        எனில் (முனை->op != ops[i]) {
            திரும்பு false;
        }
        எனில் ((முனை->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            திரும்பு false;
        }
        எனில் (i < இலக்கம்_ops - 1 && !ggml_முனை_has_n_uses(cgraph, முனை_idxs[i], 1)) {
            திரும்பு false;
        }
        எனில் (i > 0) {
            கட்டமைப்பு ggml_tensor * prev = cgraph->nodes[முனை_idxs[i - 1]];
            எனில் (முனை->src[0] != prev && முனை->src[1] != prev) {
                திரும்பு false;
            }
            எனில் (!ggml_are_same_shape(முனை, prev)) {
                திரும்பு false;
            }
        }
    }
    திரும்பு true;
}

// same as above, for sequential indices starting at node_idx
நிலையான உள்வரி bool ggml_can_fuse(மாறிலி கட்டமைப்பு ggml_cgraph * cgraph, முழு முனை_idx, மாறிலி எண்ணிக்கை ggml_op * ops, முழு இலக்கம்_ops) {
    ப்ரதிஜ்ஞா(இலக்கம்_ops < 32);

    எனில் (முனை_idx + இலக்கம்_ops > cgraph->n_nodes) {
        திரும்பு false;
    }

    முழு idxs[32];
    க்கு (முழு i = 0; i < இலக்கம்_ops; ++i) {
        idxs[i] = முனை_idx + i;
    }

    திரும்பு ggml_can_fuse_ext(cgraph, idxs, ops, இலக்கம்_ops);
}

GGML_API bool ggml_can_fuse_subgraph_ext(மாறிலி கட்டமைப்பு ggml_cgraph * cgraph,
                                         மாறிலி முழு *                முனை_idxs,
                                         முழு                        மொத்தம்,
                                         மாறிலி எண்ணிக்கை ggml_op *       ops,
                                         மாறிலி முழு *                outputs,
                                         முழு                        இலக்கம்_outputs);

// Returns true if the subgraph formed by {node_idxs} can be fused
// checks whethers all nodes which are not part of outputs can be elided
// by checking if their num_uses are confined to the subgraph
நிலையான உள்வரி bool ggml_can_fuse_subgraph(மாறிலி கட்டமைப்பு ggml_cgraph * cgraph,
                                          முழு                        முனை_idx,
                                          முழு                        மொத்தம்,
                                          மாறிலி எண்ணிக்கை ggml_op *       ops,
                                          மாறிலி முழு *                outputs,
                                          முழு                        இலக்கம்_outputs) {
    GGML_ASSERT(மொத்தம் < 32);
    எனில் (முனை_idx + மொத்தம் > cgraph->n_nodes) {
        திரும்பு false;
    }

    முழு idxs[32];

    க்கு (முழு i = 0; i < மொத்தம்; ++i) {
        idxs[i] = முனை_idx + i;
    }

    திரும்பு ggml_can_fuse_subgraph_ext(cgraph, idxs, மொத்தம், ops, outputs, இலக்கம்_outputs);
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#அந்தர்பாவ <array>
#அந்தர்பாவ <initializer_list>
#அந்தர்பாவ <vector>

// nicer C++ syntax for ggml_can_fuse
உள்வரி bool ggml_can_fuse(மாறிலி கட்டமைப்பு ggml_cgraph * cgraph, முழு முனை_idx, std::initializer_list<எண்ணிக்கை ggml_op> ops) {
    திரும்பு ggml_can_fuse(cgraph, முனை_idx, ops.begin(), (முழு)ops.size());
}

உள்வரி bool ggml_can_fuse_subgraph(மாறிலி கட்டமைப்பு ggml_cgraph *          cgraph,
                                   முழு                                 start_idx,
                                   std::initializer_list<எண்ணிக்கை ggml_op> ops,
                                   std::initializer_list<முழு>          outputs = {}) {
    திரும்பு ggml_can_fuse_subgraph(cgraph, start_idx, ops.size(), ops.begin(), outputs.begin(), outputs.size());
}

// Return true if the edges in the graph match expectations.
உள்வரி bool ggml_check_edges(மாறிலி கட்டமைப்பு ggml_cgraph *                cgraph,
                             முழு                                       start_idx,
                             std::initializer_list<std::array<முழு, 3>> edges) {
    க்கு (மாறிலி auto & edge : edges) {
        முழு dst_முனை = edge[0];
        முழு src_idx  = edge[1];
        முழு src_முனை = edge[2];
        எனில் (cgraph->nodes[start_idx + dst_முனை]->src[src_idx] != cgraph->nodes[start_idx + src_முனை]) {
            திரும்பு false;
        }
    }
    திரும்பு true;
}

// expose GGUF internals for test code
GGML_API size_t gguf_type_size(எண்ணிக்கை gguf_type type);
GGML_API வெறுமை gguf_write_to_buf(மாறிலி கட்டமைப்பு gguf_context * ctx, std::vector<int8_t> & buf, bool only_meta);
#endif // __cplusplus
