#pragma once

// GGML internal header

#अन्तर्भाव "ggml.h"
#अन्तर्भाव "gguf.h"

#अन्तर्भाव <assert.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <string.h>

#ifdef __ARM_FEATURE_SVE
#अन्तर्भाव <arm_sve.h>
#endif // __ARM_FEATURE_SVE

#जइ defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#अन्तर्भाव <arm_neon.h>
#endif

#ifdef __cplusplus
बाह्य "C" {
#endif

शून्य ggml_print_backtrace(शून्य);

uint64_t ggml_graph_next_uid(शून्य);

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
        #जइ defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
            #define static_assert(cond, सन्देशः) _Static_assert(cond, सन्देशः)
        #अण्णहा
            #define static_assert(cond, सन्देशः) संरचना global_scope_noop_trick
        #endif
    #endif
#endif

स्थिर अन्तःस्थ पूर्णाङ्क ggml_up32(पूर्णाङ्क n) {
    प्रत्यागम (n + 31) & ~31;
}

//static inline int ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

स्थिर अन्तःस्थ पूर्णाङ्क ggml_up(पूर्णाङ्क n, पूर्णाङ्क m) {
    // assert m is a power of 2
    GGML_ASSERT((m & (m - 1)) == 0);
    प्रत्यागम (n + m - 1) & ~(m - 1);
}

// TODO: move to ggml.h? (won't be able to inline)
स्थिर bool ggml_are_same_layout(ध्रुव संरचना ggml_tensor * a, ध्रुव संरचना ggml_tensor * b) {
    जइ (a->type != b->type) {
        प्रत्यागम false;
    }
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; i++) {
        जइ (a->ne[i] != b->ne[i]) {
            प्रत्यागम false;
        }
        जइ (a->nb[i] != b->nb[i]) {
            प्रत्यागम false;
        }
    }
    प्रत्यागम true;
}

स्थिर bool ggml_op_is_empty(गणना ggml_op op) {
    विकल्प (op) {
        पक्ष GGML_OP_NONE:
        पक्ष GGML_OP_RESHAPE:
        पक्ष GGML_OP_TRANSPOSE:
        पक्ष GGML_OP_VIEW:
        पक्ष GGML_OP_PERMUTE:
            प्रत्यागम true;
        स्वभाव:
            प्रत्यागम false;
    }
}

स्थिर अन्तःस्थ bool ggml_impl_is_view(ध्रुव संरचना ggml_tensor * t) {
    प्रत्यागम t->view_src != रिक्त;
}

स्थिर अन्तःस्थ भिन्न ggml_compute_softplus_f32(भिन्न निवेशः) {
    प्रत्यागम (निवेशः > 20.0f) ? निवेशः : logf(1 + expf(निवेशः));
}
//
// logging
//

GGML_ATTRIBUTE_FORMAT(2, 3)
GGML_API शून्य ggml_log_internal        (गणना ggml_log_level level, ध्रुव वर्ण * format, ...);
GGML_API शून्य ggml_log_callback_default(गणना ggml_log_level level, ध्रुव वर्ण * text, शून्य * user_data);

#define GGML_LOG(...)       ggml_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define GGML_LOG_INFO(...)  ggml_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define GGML_LOG_WARN(...)  ggml_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define GGML_LOG_ERROR(...) ggml_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GGML_LOG_DEBUG(...) ggml_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define GGML_LOG_CONT(...)  ggml_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

#define GGML_DEBUG 0

#जइ (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#अण्णहा
#define GGML_PRINT_DEBUG(...)
#endif

#जइ (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) GGML_LOG_DEBUG(__VA_ARGS__)
#अण्णहा
#define GGML_PRINT_DEBUG_5(...)
#endif

#जइ (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) GGML_LOG_DEBUG(__VA_ARGS__)
#अण्णहा
#define GGML_PRINT_DEBUG_10(...)
#endif

// tensor params

स्थिर शून्य ggml_set_op_params(संरचना ggml_tensor * tensor, ध्रुव शून्य * params, size_t params_size) {
    GGML_ASSERT(tensor != रिक्त); // silence -Warray-bounds warnings
    प्रतिज्ञा(params_size <= GGML_MAX_OP_PARAMS);
    memcpy(tensor->op_params, params, params_size);
}

स्थिर int32_t ggml_get_op_params_i32(ध्रुव संरचना ggml_tensor * tensor, uint32_t i) {
    प्रतिज्ञा(i < GGML_MAX_OP_PARAMS / परिमाण(int32_t));
    प्रत्यागम ((ध्रुव int32_t *)(tensor->op_params))[i];
}

स्थिर भिन्न ggml_get_op_params_f32(ध्रुव संरचना ggml_tensor * tensor, uint32_t i) {
    प्रतिज्ञा(i < GGML_MAX_OP_PARAMS / परिमाण(भिन्न));
    प्रत्यागम ((ध्रुव भिन्न *)(tensor->op_params))[i];
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
स्थिर शून्य ggml_set_op_params_i32(संरचना ggml_tensor * tensor, uint32_t i, int32_t मानम्) {
    प्रतिज्ञा(i < GGML_MAX_OP_PARAMS / परिमाण(int32_t));
    ((int32_t *)(tensor->op_params))[i] = मानम्;
}

स्थिर शून्य ggml_set_op_params_f32(संरचना ggml_tensor * tensor, uint32_t i, भिन्न मानम्) {
    प्रतिज्ञा(i < GGML_MAX_OP_PARAMS / परिमाण(भिन्न));
    ((भिन्न *)(tensor->op_params))[i] = मानम्;
}

संरचना ggml_map_custom1_op_params {
    ggml_custom1_op_t  fun;
    पूर्णाङ्क                n_tasks;
    शून्य             * userdata;
};

संरचना ggml_map_custom2_op_params {
    ggml_custom2_op_t   fun;
    पूर्णाङ्क                 n_tasks;
    शून्य              * userdata;
};

संरचना ggml_map_custom3_op_params {
    ggml_custom3_op_t fun;
    पूर्णाङ्क               n_tasks;
    शून्य            * userdata;
};

संरचना ggml_custom_op_params {
    ggml_custom_op_t fun;
    पूर्णाङ्क              n_tasks;
    शून्य           * userdata;
};

// bitset

प्रकारनिर्देश uint32_t ggml_bitset_t;

static_assert(परिमाण(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#define BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#define BITSET_MASK (परिमाण(ggml_bitset_t)*8 - 1)

स्थिर size_t ggml_bitset_size(size_t n) {
    प्रत्यागम (n + BITSET_MASK) >> BITSET_SHR;
}

स्थिर अन्तःस्थ bool ggml_bitset_get(ध्रुव ggml_bitset_t * bitset, size_t i) {
    प्रत्यागम !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
}

स्थिर अन्तःस्थ शून्य ggml_bitset_set(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_MASK));
}

स्थिर अन्तःस्थ शून्य ggml_bitset_clear(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_MASK));
}

// hash set

#define GGML_HASHSET_FULL ((size_t)-1)
#define GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)

संरचना ggml_hash_set {
    size_t size;
    ggml_bitset_t * used;       // whether or not the keys are in use i.e. set
    संरचना ggml_tensor ** कुञ्चिकाः; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};

संरचना ggml_hash_set ggml_hash_set_new(size_t size);
शून्य                 ggml_hash_set_free(संरचना ggml_hash_set * hash_set);

// returns the minimum size for a hash set that can hold min_sz elements
size_t ggml_hash_size(size_t min_sz);

// remove all elements from the hash set
शून्य ggml_hash_set_reset(संरचना ggml_hash_set * hash_set);

// returns true if key is in the hash set
स्थिर bool ggml_hash_contains(ध्रुव संरचना ggml_hash_set * hash_set, संरचना ggml_tensor * कुञ्चिका);

// returns GGML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
स्थिर size_t ggml_hash_find(ध्रुव संरचना ggml_hash_set * hash_set, ध्रुव संरचना ggml_tensor * कुञ्चिका);

// returns GGML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
स्थिर size_t ggml_hash_निवेशय(संरचना ggml_hash_set * hash_set, संरचना ggml_tensor * कुञ्चिका);

// return index, asserts if table is full
स्थिर size_t ggml_hash_find_or_निवेशय(संरचना ggml_hash_set * hash_set, संरचना ggml_tensor * कुञ्चिका);

// hash function for ggml_tensor
स्थिर अन्तःस्थ size_t ggml_hash(ध्रुव संरचना ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    प्रत्यागम (size_t)(uintptr_t)p >> 4;
}

स्थिर size_t ggml_hash_find(ध्रुव संरचना ggml_hash_set * hash_set, ध्रुव संरचना ggml_tensor * कुञ्चिका) {
    size_t h = ggml_hash(कुञ्चिका) % hash_set->size;

    // linear probing
    size_t i = h;
    जाव (ggml_bitset_get(hash_set->used, i) && hash_set->कुञ्चिकाः[i] != कुञ्चिका) {
        i = (i + 1) % hash_set->size;
        जइ (i == h) {
            // visited all hash table entries -> not found
            प्रत्यागम GGML_HASHSET_FULL;
        }
    }
    प्रत्यागम i;
}

स्थिर bool ggml_hash_contains(ध्रुव संरचना ggml_hash_set * hash_set, संरचना ggml_tensor * कुञ्चिका) {
    size_t i = ggml_hash_find(hash_set, कुञ्चिका);
    प्रत्यागम i != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, i);
}

स्थिर size_t ggml_hash_निवेशय(संरचना ggml_hash_set * hash_set, संरचना ggml_tensor * कुञ्चिका) {
    size_t h = ggml_hash(कुञ्चिका) % hash_set->size;

    // linear probing
    size_t i = h;
    करेहि {
        जइ (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->कुञ्चिकाः[i] = कुञ्चिका;
            प्रत्यागम i;
        }
        जइ (hash_set->कुञ्चिकाः[i] == कुञ्चिका) {
            प्रत्यागम GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_set->size;
    } जाव (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

स्थिर size_t ggml_hash_find_or_निवेशय(संरचना ggml_hash_set * hash_set, संरचना ggml_tensor * कुञ्चिका) {
    size_t h = ggml_hash(कुञ्चिका) % hash_set->size;

    // linear probing
    size_t i = h;
    करेहि {
        जइ (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->कुञ्चिकाः[i] = कुञ्चिका;
            प्रत्यागम i;
        }
        जइ (hash_set->कुञ्चिकाः[i] == कुञ्चिका) {
            प्रत्यागम i;
        }
        i = (i + 1) % hash_set->size;
    } जाव (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

// computation graph

गणना ggml_cgraph_eval_order {
    GGML_CGRAPH_EVAL_ORDER_वामः_TO_दक्षिणः = 0,
    GGML_CGRAPH_EVAL_ORDER_दक्षिणः_TO_वामः,
    GGML_CGRAPH_EVAL_ORDER_गणनम्
};

संरचना ggml_cgraph {
    पूर्णाङ्क size;    // maximum number of nodes/leafs/grads/grad_accs
    पूर्णाङ्क n_nodes; // number of nodes currently in use
    पूर्णाङ्क n_leafs; // number of leafs currently in use

    संरचना ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
    संरचना ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
    संरचना ggml_tensor ** grad_accs; // accumulators for node gradients
    संरचना ggml_tensor ** leafs;     // tensors with constant data
    int32_t             * use_counts;// number of uses of each tensor, indexed by hash table slot

    संरचना ggml_hash_set visited_hash_set;

    गणना ggml_cgraph_eval_order order;

    // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
    // a value of 0 means it is not set and should be ignored
    uint64_t uid;
};

// returns a slice of cgraph with nodes [i0, i1)
// the slice does not have leafs or gradients
// if you need the gradients, get them from the original graph
संरचना ggml_cgraph ggml_graph_view(संरचना ggml_cgraph * cgraph, पूर्णाङ्क i0, पूर्णाङ्क i1);

// ggml-alloc.c: true if the operation can reuse memory from its sources
GGML_API bool ggml_op_can_inplace(गणना ggml_op op);


// Memory allocation

GGML_API शून्य * ggml_aligned_malloc(size_t size);
GGML_API शून्य ggml_aligned_free(शून्य * ptr, size_t size);

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

स्थिर अन्तःस्थ भिन्न fp32_from_bits(uint32_t w) {
    सङ्घ {
        uint32_t as_bits;
        भिन्न as_मानम्;
    } fp32;
    fp32.as_bits = w;
    प्रत्यागम fp32.as_मानम्;
}

स्थिर अन्तःस्थ uint32_t fp32_to_bits(भिन्न f) {
    सङ्घ {
        भिन्न as_मानम्;
        uint32_t as_bits;
    } fp32;
    fp32.as_मानम् = f;
    प्रत्यागम fp32.as_bits;
}

स्थिर अन्तःस्थ भिन्न ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    ध्रुव uint32_t w = (uint32_t) h << 16;
    ध्रुव uint32_t sign = w & UINT32_C(0x80000000);
    ध्रुव uint32_t two_w = w + w;

    ध्रुव uint32_t exp_offset = UINT32_C(0xE0) << 23;
#जइ (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    ध्रुव भिन्न exp_scale = 0x1.0p-112f;
#अण्णहा
    ध्रुव भिन्न exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    ध्रुव भिन्न normalized_मानम् = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    ध्रुव uint32_t magic_mask = UINT32_C(126) << 23;
    ध्रुव भिन्न magic_bias = 0.5f;
    ध्रुव भिन्न denormalized_मानम् = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    ध्रुव uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    ध्रुव uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_मानम्) : fp32_to_bits(normalized_मानम्));
    प्रत्यागम fp32_from_bits(result);
}

स्थिर अन्तःस्थ ggml_fp16_t ggml_compute_fp32_to_fp16(भिन्न f) {
#जइ (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    ध्रुव भिन्न scale_to_inf = 0x1.0p+112f;
    ध्रुव भिन्न scale_to_zero = 0x1.0p-110f;
#अण्णहा
    ध्रुव भिन्न scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    ध्रुव भिन्न scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    भिन्न base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    ध्रुव uint32_t w = fp32_to_bits(f);
    ध्रुव uint32_t shl1_w = w + w;
    ध्रुव uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    जइ (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    ध्रुव uint32_t bits = fp32_to_bits(base);
    ध्रुव uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    ध्रुव uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    ध्रुव uint32_t nonsign = exp_bits + mantissa_bits;
    प्रत्यागम (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

स्थिर अन्तःस्थ भिन्न ggml_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;  // Stores the raw bit representation of the float

    // Handle special case for minimum exponent (denormalized float)
    जइ (x == 0) {
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
    अण्णहा {
        // Construct normalized float by shifting exponent into position:
        // - Exponent field: 8 bits (positions 30-23)
        // - Mantissa: 0 (implicit leading 1)
        // Value = 2^(x - 127)
        bits = (uint32_t) x << 23;
    }

    भिन्न result;  // Final float value
                   // Safely reinterpret bit pattern as float without type-punning issues
    memcpy(&result, &bits, परिमाण(भिन्न));
    प्रत्यागम result;
}

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
स्थिर अन्तःस्थ भिन्न ggml_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;

    // For x < 2: use precomputed denormal patterns
    जइ (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        bits = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    अण्णहा {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        bits = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    भिन्न result;
    memcpy(&result, &bits, परिमाण(भिन्न));
    प्रत्यागम result;
}

#define GGML_E8M0_TO_FP32(x) ggml_e8m0_to_fp32(x)
#define GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
स्थिर अन्तःस्थ भिन्न ggml_ue4m3_to_fp32(uint8_t x) {
    जइ (x == 0 || x == 0x7F) {
        प्रत्यागम 0.0f;
    }
    पूर्णाङ्क   exp = (x >> 3) & 0xF;
    पूर्णाङ्क   man = x & 0x7;
    भिन्न raw;
    जइ (exp == 0) {
        raw = ldexpf((भिन्न) man, -9);
    } अण्णहा {
        raw = ldexpf(1.0f + (भिन्न) man / 8.0f, exp - 7);
    }
    प्रत्यागम raw * 0.5f;
}

स्थिर अन्तःस्थ uint8_t ggml_fp32_to_ue4m3(भिन्न x) {
    जइ (!(x > 0.0f)) {
        प्रत्यागम 0;
    }
    जइ (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t bits;
    memcpy(&bits, &x, 4);
    पूर्णाङ्क fp32_exp  = ((bits >> 23) & 0xFF) - 127;
    पूर्णाङ्क fp32_man  = (bits >> 20) & 0x7;
    पूर्णाङ्क ue4m3_exp = fp32_exp + 7;
    जइ (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        पूर्णाङ्क man = (पूर्णाङ्क) (x * 512.0f + 0.5f);
        जइ (man > 7) {
            man = 7;
        }
        जइ (man < 1) {
            प्रत्यागम 0;
        }
        प्रत्यागम (uint8_t) man;
    }
    जइ (ue4m3_exp >= 15) {
        प्रत्यागम 0x7E;
    }
    पूर्णाङ्क round_bit = (bits >> 19) & 1;
    पूर्णाङ्क ue4m3_man = fp32_man + round_bit;
    जइ (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        जइ (ue4m3_exp >= 15) {
            प्रत्यागम 0x7E;
        }
    }
    प्रत्यागम (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
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
स्थिर अन्तःस्थ भिन्न ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    सङ्घ {
        भिन्न f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    प्रत्यागम u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
स्थिर अन्तःस्थ ggml_bf16_t ggml_compute_fp32_to_bf16(भिन्न s) {
    ggml_bf16_t h;
    सङ्घ {
        भिन्न f;
        uint32_t i;
    } u;
    u.f = s;
    जइ ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.bits = (u.i >> 16) | 64; /* force to quiet */
        प्रत्यागम h;
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    प्रत्यागम h;
}

#define GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)
#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)

स्थिर अन्तःस्थ int32_t ggml_ग्रन्थिः_get_use_गणनम्(ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    ध्रुव संरचना ggml_tensor * ग्रन्थिः = cgraph->nodes[ग्रन्थिः_idx];

    size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, ग्रन्थिः);
    जइ (!ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        प्रत्यागम 0;
    }
    प्रत्यागम cgraph->use_counts[hash_pos];
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
स्थिर अन्तःस्थ bool ggml_ग्रन्थिः_has_n_uses(ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, int32_t n_uses) {
    ध्रुव संरचना ggml_tensor * ग्रन्थिः = cgraph->nodes[ग्रन्थिः_idx];

    // check the use count against how many we're replacing
    जइ (ggml_ग्रन्थिः_get_use_गणनम्(cgraph, ग्रन्थिः_idx) != n_uses) {
        प्रत्यागम false;
    }

    // if node is a view, some other node might be using the intermediate result
    // via the view source.
    जइ (ग्रन्थिः->view_src) {
        प्रत्यागम false;
    }

    // If the user requested output for the node, can't fuse
    जइ (ग्रन्थिः->flags & GGML_TENSOR_FLAG_OUTPUT) {
        प्रत्यागम false;
    }

    प्रत्यागम true;
}

// Returns true if nodes with indices { node_idxs } are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
स्थिर अन्तःस्थ bool ggml_can_fuse_ext(ध्रुव संरचना ggml_cgraph * cgraph, ध्रुव पूर्णाङ्क * ग्रन्थिः_idxs, ध्रुव गणना ggml_op * ops, पूर्णाङ्क अङ्कः_ops) {
    प्रति (पूर्णाङ्क i = 0; i < अङ्कः_ops; ++i) {
        जइ (ग्रन्थिः_idxs[i] >= cgraph->n_nodes) {
            प्रत्यागम false;
        }

        संरचना ggml_tensor * ग्रन्थिः = cgraph->nodes[ग्रन्थिः_idxs[i]];
        जइ (ग्रन्थिः->op != ops[i]) {
            प्रत्यागम false;
        }
        जइ ((ग्रन्थिः->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            प्रत्यागम false;
        }
        जइ (i < अङ्कः_ops - 1 && !ggml_ग्रन्थिः_has_n_uses(cgraph, ग्रन्थिः_idxs[i], 1)) {
            प्रत्यागम false;
        }
        जइ (i > 0) {
            संरचना ggml_tensor * prev = cgraph->nodes[ग्रन्थिः_idxs[i - 1]];
            जइ (ग्रन्थिः->src[0] != prev && ग्रन्थिः->src[1] != prev) {
                प्रत्यागम false;
            }
            जइ (!ggml_are_same_shape(ग्रन्थिः, prev)) {
                प्रत्यागम false;
            }
        }
    }
    प्रत्यागम true;
}

// same as above, for sequential indices starting at node_idx
स्थिर अन्तःस्थ bool ggml_can_fuse(ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, ध्रुव गणना ggml_op * ops, पूर्णाङ्क अङ्कः_ops) {
    प्रतिज्ञा(अङ्कः_ops < 32);

    जइ (ग्रन्थिः_idx + अङ्कः_ops > cgraph->n_nodes) {
        प्रत्यागम false;
    }

    पूर्णाङ्क idxs[32];
    प्रति (पूर्णाङ्क i = 0; i < अङ्कः_ops; ++i) {
        idxs[i] = ग्रन्थिः_idx + i;
    }

    प्रत्यागम ggml_can_fuse_ext(cgraph, idxs, ops, अङ्कः_ops);
}

GGML_API bool ggml_can_fuse_subgraph_ext(ध्रुव संरचना ggml_cgraph * cgraph,
                                         ध्रुव पूर्णाङ्क *                ग्रन्थिः_idxs,
                                         पूर्णाङ्क                        गणनम्,
                                         ध्रुव गणना ggml_op *       ops,
                                         ध्रुव पूर्णाङ्क *                outputs,
                                         पूर्णाङ्क                        अङ्कः_outputs);

// Returns true if the subgraph formed by {node_idxs} can be fused
// checks whethers all nodes which are not part of outputs can be elided
// by checking if their num_uses are confined to the subgraph
स्थिर अन्तःस्थ bool ggml_can_fuse_subgraph(ध्रुव संरचना ggml_cgraph * cgraph,
                                          पूर्णाङ्क                        ग्रन्थिः_idx,
                                          पूर्णाङ्क                        गणनम्,
                                          ध्रुव गणना ggml_op *       ops,
                                          ध्रुव पूर्णाङ्क *                outputs,
                                          पूर्णाङ्क                        अङ्कः_outputs) {
    GGML_ASSERT(गणनम् < 32);
    जइ (ग्रन्थिः_idx + गणनम् > cgraph->n_nodes) {
        प्रत्यागम false;
    }

    पूर्णाङ्क idxs[32];

    प्रति (पूर्णाङ्क i = 0; i < गणनम्; ++i) {
        idxs[i] = ग्रन्थिः_idx + i;
    }

    प्रत्यागम ggml_can_fuse_subgraph_ext(cgraph, idxs, गणनम्, ops, outputs, अङ्कः_outputs);
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#अन्तर्भाव <array>
#अन्तर्भाव <initializer_list>
#अन्तर्भाव <vector>

// nicer C++ syntax for ggml_can_fuse
अन्तःस्थ bool ggml_can_fuse(ध्रुव संरचना ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, std::initializer_list<गणना ggml_op> ops) {
    प्रत्यागम ggml_can_fuse(cgraph, ग्रन्थिः_idx, ops.begin(), (पूर्णाङ्क)ops.size());
}

अन्तःस्थ bool ggml_can_fuse_subgraph(ध्रुव संरचना ggml_cgraph *          cgraph,
                                   पूर्णाङ्क                                 start_idx,
                                   std::initializer_list<गणना ggml_op> ops,
                                   std::initializer_list<पूर्णाङ्क>          outputs = {}) {
    प्रत्यागम ggml_can_fuse_subgraph(cgraph, start_idx, ops.size(), ops.begin(), outputs.begin(), outputs.size());
}

// Return true if the edges in the graph match expectations.
अन्तःस्थ bool ggml_check_edges(ध्रुव संरचना ggml_cgraph *                cgraph,
                             पूर्णाङ्क                                       start_idx,
                             std::initializer_list<std::array<पूर्णाङ्क, 3>> edges) {
    प्रति (ध्रुव auto & edge : edges) {
        पूर्णाङ्क dst_ग्रन्थिः = edge[0];
        पूर्णाङ्क src_idx  = edge[1];
        पूर्णाङ्क src_ग्रन्थिः = edge[2];
        जइ (cgraph->nodes[start_idx + dst_ग्रन्थिः]->src[src_idx] != cgraph->nodes[start_idx + src_ग्रन्थिः]) {
            प्रत्यागम false;
        }
    }
    प्रत्यागम true;
}

// expose GGUF internals for test code
GGML_API size_t gguf_type_size(गणना gguf_type type);
GGML_API शून्य gguf_write_to_buf(ध्रुव संरचना gguf_context * ctx, std::vector<int8_t> & buf, bool only_meta);
#endif // __cplusplus
