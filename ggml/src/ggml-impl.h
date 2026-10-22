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

#जर defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
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
        #जर defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
            #define static_assert(cond, संदेश) _Static_assert(cond, संदेश)
        #अन्यथा
            #define static_assert(cond, संदेश) संरचना global_scope_noop_trick
        #endif
    #endif
#endif

स्थिर ओळीत पूर्णांक ggml_up32(पूर्णांक n) {
    परत (n + 31) & ~31;
}

//static inline int ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

स्थिर ओळीत पूर्णांक ggml_up(पूर्णांक n, पूर्णांक m) {
    // assert m is a power of 2
    GGML_ASSERT((m & (m - 1)) == 0);
    परत (n + m - 1) & ~(m - 1);
}

// TODO: move to ggml.h? (won't be able to inline)
स्थिर bool ggml_are_same_layout(स्थिरांक संरचना ggml_tensor * a, स्थिरांक संरचना ggml_tensor * b) {
    जर (a->type != b->type) {
        परत false;
    }
    साठी (पूर्णांक i = 0; i < GGML_MAX_DIMS; i++) {
        जर (a->ne[i] != b->ne[i]) {
            परत false;
        }
        जर (a->nb[i] != b->nb[i]) {
            परत false;
        }
    }
    परत true;
}

स्थिर bool ggml_op_is_empty(गणना ggml_op op) {
    निवड (op) {
        स्थिती GGML_OP_NONE:
        स्थिती GGML_OP_RESHAPE:
        स्थिती GGML_OP_TRANSPOSE:
        स्थिती GGML_OP_VIEW:
        स्थिती GGML_OP_PERMUTE:
            परत true;
        पूर्वनिर्धारित:
            परत false;
    }
}

स्थिर ओळीत bool ggml_impl_is_view(स्थिरांक संरचना ggml_tensor * t) {
    परत t->view_src != रिक्त;
}

स्थिर ओळीत दशांश ggml_compute_softplus_f32(दशांश आदान) {
    परत (आदान > 20.0f) ? आदान : logf(1 + expf(आदान));
}
//
// logging
//

GGML_ATTRIBUTE_FORMAT(2, 3)
GGML_API शून्य ggml_log_internal        (गणना ggml_log_level level, स्थिरांक अक्षर * format, ...);
GGML_API शून्य ggml_log_callback_default(गणना ggml_log_level level, स्थिरांक अक्षर * text, शून्य * user_data);

#define GGML_LOG(...)       ggml_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define GGML_LOG_INFO(...)  ggml_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define GGML_LOG_WARN(...)  ggml_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define GGML_LOG_ERROR(...) ggml_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GGML_LOG_DEBUG(...) ggml_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define GGML_LOG_CONT(...)  ggml_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

#define GGML_DEBUG 0

#जर (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#अन्यथा
#define GGML_PRINT_DEBUG(...)
#endif

#जर (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) GGML_LOG_DEBUG(__VA_ARGS__)
#अन्यथा
#define GGML_PRINT_DEBUG_5(...)
#endif

#जर (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) GGML_LOG_DEBUG(__VA_ARGS__)
#अन्यथा
#define GGML_PRINT_DEBUG_10(...)
#endif

// tensor params

स्थिर शून्य ggml_set_op_params(संरचना ggml_tensor * tensor, स्थिरांक शून्य * params, माप_प्रकार params_size) {
    GGML_ASSERT(tensor != रिक्त); // silence -Warray-bounds warnings
    प्रतिज्ञा(params_size <= GGML_MAX_OP_PARAMS);
    स_नकल(tensor->op_params, params, params_size);
}

स्थिर int32_t ggml_get_op_params_i32(स्थिरांक संरचना ggml_tensor * tensor, uint32_t i) {
    प्रतिज्ञा(i < GGML_MAX_OP_PARAMS / आकार(int32_t));
    परत ((स्थिरांक int32_t *)(tensor->op_params))[i];
}

स्थिर दशांश ggml_get_op_params_f32(स्थिरांक संरचना ggml_tensor * tensor, uint32_t i) {
    प्रतिज्ञा(i < GGML_MAX_OP_PARAMS / आकार(दशांश));
    परत ((स्थिरांक दशांश *)(tensor->op_params))[i];
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
स्थिर शून्य ggml_set_op_params_i32(संरचना ggml_tensor * tensor, uint32_t i, int32_t मूल्य) {
    प्रतिज्ञा(i < GGML_MAX_OP_PARAMS / आकार(int32_t));
    ((int32_t *)(tensor->op_params))[i] = मूल्य;
}

स्थिर शून्य ggml_set_op_params_f32(संरचना ggml_tensor * tensor, uint32_t i, दशांश मूल्य) {
    प्रतिज्ञा(i < GGML_MAX_OP_PARAMS / आकार(दशांश));
    ((दशांश *)(tensor->op_params))[i] = मूल्य;
}

संरचना ggml_map_custom1_op_params {
    ggml_custom1_op_t  fun;
    पूर्णांक                n_tasks;
    शून्य             * userdata;
};

संरचना ggml_map_custom2_op_params {
    ggml_custom2_op_t   fun;
    पूर्णांक                 n_tasks;
    शून्य              * userdata;
};

संरचना ggml_map_custom3_op_params {
    ggml_custom3_op_t fun;
    पूर्णांक               n_tasks;
    शून्य            * userdata;
};

संरचना ggml_custom_op_params {
    ggml_custom_op_t fun;
    पूर्णांक              n_tasks;
    शून्य           * userdata;
};

// bitset

प्रकारव्याख्या uint32_t ggml_bitset_t;

static_assert(आकार(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#define BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#define BITSET_MASK (आकार(ggml_bitset_t)*8 - 1)

स्थिर माप_प्रकार ggml_bitset_size(माप_प्रकार n) {
    परत (n + BITSET_MASK) >> BITSET_SHR;
}

स्थिर ओळीत bool ggml_bitset_get(स्थिरांक ggml_bitset_t * bitset, माप_प्रकार i) {
    परत !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
}

स्थिर ओळीत शून्य ggml_bitset_set(ggml_bitset_t * bitset, माप_प्रकार i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_MASK));
}

स्थिर ओळीत शून्य ggml_bitset_clear(ggml_bitset_t * bitset, माप_प्रकार i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_MASK));
}

// hash set

#define GGML_HASHSET_FULL ((माप_प्रकार)-1)
#define GGML_HASHSET_ALREADY_EXISTS ((माप_प्रकार)-2)

संरचना ggml_hash_set {
    माप_प्रकार size;
    ggml_bitset_t * used;       // whether or not the keys are in use i.e. set
    संरचना ggml_tensor ** किल्ल्या; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};

संरचना ggml_hash_set ggml_hash_set_new(माप_प्रकार size);
शून्य                 ggml_hash_set_free(संरचना ggml_hash_set * hash_set);

// returns the minimum size for a hash set that can hold min_sz elements
माप_प्रकार ggml_hash_size(माप_प्रकार min_sz);

// remove all elements from the hash set
शून्य ggml_hash_set_reset(संरचना ggml_hash_set * hash_set);

// returns true if key is in the hash set
स्थिर bool ggml_hash_contains(स्थिरांक संरचना ggml_hash_set * hash_set, संरचना ggml_tensor * किल्ली);

// returns GGML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
स्थिर माप_प्रकार ggml_hash_find(स्थिरांक संरचना ggml_hash_set * hash_set, स्थिरांक संरचना ggml_tensor * किल्ली);

// returns GGML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
स्थिर माप_प्रकार ggml_hash_घाला(संरचना ggml_hash_set * hash_set, संरचना ggml_tensor * किल्ली);

// return index, asserts if table is full
स्थिर माप_प्रकार ggml_hash_find_or_घाला(संरचना ggml_hash_set * hash_set, संरचना ggml_tensor * किल्ली);

// hash function for ggml_tensor
स्थिर ओळीत माप_प्रकार ggml_hash(स्थिरांक संरचना ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    परत (माप_प्रकार)(uintptr_t)p >> 4;
}

स्थिर माप_प्रकार ggml_hash_find(स्थिरांक संरचना ggml_hash_set * hash_set, स्थिरांक संरचना ggml_tensor * किल्ली) {
    माप_प्रकार h = ggml_hash(किल्ली) % hash_set->size;

    // linear probing
    माप_प्रकार i = h;
    पर्यंत (ggml_bitset_get(hash_set->used, i) && hash_set->किल्ल्या[i] != किल्ली) {
        i = (i + 1) % hash_set->size;
        जर (i == h) {
            // visited all hash table entries -> not found
            परत GGML_HASHSET_FULL;
        }
    }
    परत i;
}

स्थिर bool ggml_hash_contains(स्थिरांक संरचना ggml_hash_set * hash_set, संरचना ggml_tensor * किल्ली) {
    माप_प्रकार i = ggml_hash_find(hash_set, किल्ली);
    परत i != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, i);
}

स्थिर माप_प्रकार ggml_hash_घाला(संरचना ggml_hash_set * hash_set, संरचना ggml_tensor * किल्ली) {
    माप_प्रकार h = ggml_hash(किल्ली) % hash_set->size;

    // linear probing
    माप_प्रकार i = h;
    करा {
        जर (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->किल्ल्या[i] = किल्ली;
            परत i;
        }
        जर (hash_set->किल्ल्या[i] == किल्ली) {
            परत GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_set->size;
    } पर्यंत (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

स्थिर माप_प्रकार ggml_hash_find_or_घाला(संरचना ggml_hash_set * hash_set, संरचना ggml_tensor * किल्ली) {
    माप_प्रकार h = ggml_hash(किल्ली) % hash_set->size;

    // linear probing
    माप_प्रकार i = h;
    करा {
        जर (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->किल्ल्या[i] = किल्ली;
            परत i;
        }
        जर (hash_set->किल्ल्या[i] == किल्ली) {
            परत i;
        }
        i = (i + 1) % hash_set->size;
    } पर्यंत (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

// computation graph

गणना ggml_cgraph_eval_order {
    GGML_CGRAPH_EVAL_ORDER_डावा_TO_उजवा = 0,
    GGML_CGRAPH_EVAL_ORDER_उजवा_TO_डावा,
    GGML_CGRAPH_EVAL_ORDER_मोजणी
};

संरचना ggml_cgraph {
    पूर्णांक size;    // maximum number of nodes/leafs/grads/grad_accs
    पूर्णांक n_nodes; // number of nodes currently in use
    पूर्णांक n_leafs; // number of leafs currently in use

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
संरचना ggml_cgraph ggml_graph_view(संरचना ggml_cgraph * cgraph, पूर्णांक i0, पूर्णांक i1);

// ggml-alloc.c: true if the operation can reuse memory from its sources
GGML_API bool ggml_op_can_inplace(गणना ggml_op op);


// Memory allocation

GGML_API शून्य * ggml_aligned_malloc(माप_प्रकार size);
GGML_API शून्य ggml_aligned_free(शून्य * ptr, माप_प्रकार size);

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

स्थिर ओळीत दशांश fp32_from_bits(uint32_t w) {
    संघ {
        uint32_t as_bits;
        दशांश as_मूल्य;
    } fp32;
    fp32.as_bits = w;
    परत fp32.as_मूल्य;
}

स्थिर ओळीत uint32_t fp32_to_bits(दशांश f) {
    संघ {
        दशांश as_मूल्य;
        uint32_t as_bits;
    } fp32;
    fp32.as_मूल्य = f;
    परत fp32.as_bits;
}

स्थिर ओळीत दशांश ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    स्थिरांक uint32_t w = (uint32_t) h << 16;
    स्थिरांक uint32_t sign = w & UINT32_C(0x80000000);
    स्थिरांक uint32_t two_w = w + w;

    स्थिरांक uint32_t exp_offset = UINT32_C(0xE0) << 23;
#जर (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    स्थिरांक दशांश exp_scale = 0x1.0p-112f;
#अन्यथा
    स्थिरांक दशांश exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    स्थिरांक दशांश normalized_मूल्य = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    स्थिरांक uint32_t magic_mask = UINT32_C(126) << 23;
    स्थिरांक दशांश magic_bias = 0.5f;
    स्थिरांक दशांश denormalized_मूल्य = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    स्थिरांक uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    स्थिरांक uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_मूल्य) : fp32_to_bits(normalized_मूल्य));
    परत fp32_from_bits(result);
}

स्थिर ओळीत ggml_fp16_t ggml_compute_fp32_to_fp16(दशांश f) {
#जर (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    स्थिरांक दशांश scale_to_inf = 0x1.0p+112f;
    स्थिरांक दशांश scale_to_zero = 0x1.0p-110f;
#अन्यथा
    स्थिरांक दशांश scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    स्थिरांक दशांश scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    दशांश base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    स्थिरांक uint32_t w = fp32_to_bits(f);
    स्थिरांक uint32_t shl1_w = w + w;
    स्थिरांक uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    जर (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    स्थिरांक uint32_t bits = fp32_to_bits(base);
    स्थिरांक uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    स्थिरांक uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    स्थिरांक uint32_t nonsign = exp_bits + mantissa_bits;
    परत (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

स्थिर ओळीत दशांश ggml_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;  // Stores the raw bit representation of the float

    // Handle special case for minimum exponent (denormalized float)
    जर (x == 0) {
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
    अन्यथा {
        // Construct normalized float by shifting exponent into position:
        // - Exponent field: 8 bits (positions 30-23)
        // - Mantissa: 0 (implicit leading 1)
        // Value = 2^(x - 127)
        bits = (uint32_t) x << 23;
    }

    दशांश result;  // Final float value
                   // Safely reinterpret bit pattern as float without type-punning issues
    स_नकल(&result, &bits, आकार(दशांश));
    परत result;
}

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
स्थिर ओळीत दशांश ggml_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;

    // For x < 2: use precomputed denormal patterns
    जर (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        bits = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    अन्यथा {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        bits = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    दशांश result;
    स_नकल(&result, &bits, आकार(दशांश));
    परत result;
}

#define GGML_E8M0_TO_FP32(x) ggml_e8m0_to_fp32(x)
#define GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
स्थिर ओळीत दशांश ggml_ue4m3_to_fp32(uint8_t x) {
    जर (x == 0 || x == 0x7F) {
        परत 0.0f;
    }
    पूर्णांक   exp = (x >> 3) & 0xF;
    पूर्णांक   man = x & 0x7;
    दशांश raw;
    जर (exp == 0) {
        raw = ldexpf((दशांश) man, -9);
    } अन्यथा {
        raw = ldexpf(1.0f + (दशांश) man / 8.0f, exp - 7);
    }
    परत raw * 0.5f;
}

स्थिर ओळीत uint8_t ggml_fp32_to_ue4m3(दशांश x) {
    जर (!(x > 0.0f)) {
        परत 0;
    }
    जर (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t bits;
    स_नकल(&bits, &x, 4);
    पूर्णांक fp32_exp  = ((bits >> 23) & 0xFF) - 127;
    पूर्णांक fp32_man  = (bits >> 20) & 0x7;
    पूर्णांक ue4m3_exp = fp32_exp + 7;
    जर (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        पूर्णांक man = (पूर्णांक) (x * 512.0f + 0.5f);
        जर (man > 7) {
            man = 7;
        }
        जर (man < 1) {
            परत 0;
        }
        परत (uint8_t) man;
    }
    जर (ue4m3_exp >= 15) {
        परत 0x7E;
    }
    पूर्णांक round_bit = (bits >> 19) & 1;
    पूर्णांक ue4m3_man = fp32_man + round_bit;
    जर (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        जर (ue4m3_exp >= 15) {
            परत 0x7E;
        }
    }
    परत (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
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
स्थिर ओळीत दशांश ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    संघ {
        दशांश f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    परत u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
स्थिर ओळीत ggml_bf16_t ggml_compute_fp32_to_bf16(दशांश s) {
    ggml_bf16_t h;
    संघ {
        दशांश f;
        uint32_t i;
    } u;
    u.f = s;
    जर ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.bits = (u.i >> 16) | 64; /* force to quiet */
        परत h;
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    परत h;
}

#define GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)
#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)

स्थिर ओळीत int32_t ggml_गाठ_get_use_मोजणी(स्थिरांक संरचना ggml_cgraph * cgraph, पूर्णांक गाठ_idx) {
    स्थिरांक संरचना ggml_tensor * गाठ = cgraph->nodes[गाठ_idx];

    माप_प्रकार hash_pos = ggml_hash_find(&cgraph->visited_hash_set, गाठ);
    जर (!ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        परत 0;
    }
    परत cgraph->use_counts[hash_pos];
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
स्थिर ओळीत bool ggml_गाठ_has_n_uses(स्थिरांक संरचना ggml_cgraph * cgraph, पूर्णांक गाठ_idx, int32_t n_uses) {
    स्थिरांक संरचना ggml_tensor * गाठ = cgraph->nodes[गाठ_idx];

    // check the use count against how many we're replacing
    जर (ggml_गाठ_get_use_मोजणी(cgraph, गाठ_idx) != n_uses) {
        परत false;
    }

    // if node is a view, some other node might be using the intermediate result
    // via the view source.
    जर (गाठ->view_src) {
        परत false;
    }

    // If the user requested output for the node, can't fuse
    जर (गाठ->flags & GGML_TENSOR_FLAG_OUTPUT) {
        परत false;
    }

    परत true;
}

// Returns true if nodes with indices { node_idxs } are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
स्थिर ओळीत bool ggml_can_fuse_ext(स्थिरांक संरचना ggml_cgraph * cgraph, स्थिरांक पूर्णांक * गाठ_idxs, स्थिरांक गणना ggml_op * ops, पूर्णांक अंक_ops) {
    साठी (पूर्णांक i = 0; i < अंक_ops; ++i) {
        जर (गाठ_idxs[i] >= cgraph->n_nodes) {
            परत false;
        }

        संरचना ggml_tensor * गाठ = cgraph->nodes[गाठ_idxs[i]];
        जर (गाठ->op != ops[i]) {
            परत false;
        }
        जर ((गाठ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            परत false;
        }
        जर (i < अंक_ops - 1 && !ggml_गाठ_has_n_uses(cgraph, गाठ_idxs[i], 1)) {
            परत false;
        }
        जर (i > 0) {
            संरचना ggml_tensor * prev = cgraph->nodes[गाठ_idxs[i - 1]];
            जर (गाठ->src[0] != prev && गाठ->src[1] != prev) {
                परत false;
            }
            जर (!ggml_are_same_shape(गाठ, prev)) {
                परत false;
            }
        }
    }
    परत true;
}

// same as above, for sequential indices starting at node_idx
स्थिर ओळीत bool ggml_can_fuse(स्थिरांक संरचना ggml_cgraph * cgraph, पूर्णांक गाठ_idx, स्थिरांक गणना ggml_op * ops, पूर्णांक अंक_ops) {
    प्रतिज्ञा(अंक_ops < 32);

    जर (गाठ_idx + अंक_ops > cgraph->n_nodes) {
        परत false;
    }

    पूर्णांक idxs[32];
    साठी (पूर्णांक i = 0; i < अंक_ops; ++i) {
        idxs[i] = गाठ_idx + i;
    }

    परत ggml_can_fuse_ext(cgraph, idxs, ops, अंक_ops);
}

GGML_API bool ggml_can_fuse_subgraph_ext(स्थिरांक संरचना ggml_cgraph * cgraph,
                                         स्थिरांक पूर्णांक *                गाठ_idxs,
                                         पूर्णांक                        मोजणी,
                                         स्थिरांक गणना ggml_op *       ops,
                                         स्थिरांक पूर्णांक *                outputs,
                                         पूर्णांक                        अंक_outputs);

// Returns true if the subgraph formed by {node_idxs} can be fused
// checks whethers all nodes which are not part of outputs can be elided
// by checking if their num_uses are confined to the subgraph
स्थिर ओळीत bool ggml_can_fuse_subgraph(स्थिरांक संरचना ggml_cgraph * cgraph,
                                          पूर्णांक                        गाठ_idx,
                                          पूर्णांक                        मोजणी,
                                          स्थिरांक गणना ggml_op *       ops,
                                          स्थिरांक पूर्णांक *                outputs,
                                          पूर्णांक                        अंक_outputs) {
    GGML_ASSERT(मोजणी < 32);
    जर (गाठ_idx + मोजणी > cgraph->n_nodes) {
        परत false;
    }

    पूर्णांक idxs[32];

    साठी (पूर्णांक i = 0; i < मोजणी; ++i) {
        idxs[i] = गाठ_idx + i;
    }

    परत ggml_can_fuse_subgraph_ext(cgraph, idxs, मोजणी, ops, outputs, अंक_outputs);
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#अन्तर्भाव <array>
#अन्तर्भाव <initializer_list>
#अन्तर्भाव <vector>

// nicer C++ syntax for ggml_can_fuse
ओळीत bool ggml_can_fuse(स्थिरांक संरचना ggml_cgraph * cgraph, पूर्णांक गाठ_idx, std::initializer_list<गणना ggml_op> ops) {
    परत ggml_can_fuse(cgraph, गाठ_idx, ops.begin(), (पूर्णांक)ops.size());
}

ओळीत bool ggml_can_fuse_subgraph(स्थिरांक संरचना ggml_cgraph *          cgraph,
                                   पूर्णांक                                 start_idx,
                                   std::initializer_list<गणना ggml_op> ops,
                                   std::initializer_list<पूर्णांक>          outputs = {}) {
    परत ggml_can_fuse_subgraph(cgraph, start_idx, ops.size(), ops.begin(), outputs.begin(), outputs.size());
}

// Return true if the edges in the graph match expectations.
ओळीत bool ggml_check_edges(स्थिरांक संरचना ggml_cgraph *                cgraph,
                             पूर्णांक                                       start_idx,
                             std::initializer_list<std::array<पूर्णांक, 3>> edges) {
    साठी (स्थिरांक स्वतः & edge : edges) {
        पूर्णांक dst_गाठ = edge[0];
        पूर्णांक src_idx  = edge[1];
        पूर्णांक src_गाठ = edge[2];
        जर (cgraph->nodes[start_idx + dst_गाठ]->src[src_idx] != cgraph->nodes[start_idx + src_गाठ]) {
            परत false;
        }
    }
    परत true;
}

// expose GGUF internals for test code
GGML_API माप_प्रकार gguf_type_size(गणना gguf_type type);
GGML_API शून्य gguf_write_to_buf(स्थिरांक संरचना gguf_context * ctx, std::vector<int8_t> & buf, bool only_meta);
#endif // __cplusplus
