#pragma once

// GGML internal header

#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "gguf.h"

#অন্তর্ভাব <assert.h>
#অন্তর্ভাব <math.h>
#অন্তর্ভাব <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <string.h>

#ifdef __ARM_FEATURE_SVE
#অন্তর্ভাব <arm_sve.h>
#endif // __ARM_FEATURE_SVE

#যদি defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#অন্তর্ভাব <arm_neon.h>
#endif

#ifdef __cplusplus
বাহ্যিক "C" {
#endif

শূন্য ggml_print_backtrace(শূন্য);

uint64_t ggml_graph_পরবর্তী_uid(শূন্য);

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
        #যদি defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
            #define static_assert(cond, বার্তা) _Static_assert(cond, বার্তা)
        #অন্যথা
            #define static_assert(cond, বার্তা) কাঠামো global_scope_noop_trick
        #endif
    #endif
#endif

স্থির অন্তঃরেখ পূর্ণসংখ্যা ggml_up32(পূর্ণসংখ্যা n) {
    ফেরত (n + 31) & ~31;
}

//static inline int ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

স্থির অন্তঃরেখ পূর্ণসংখ্যা ggml_up(পূর্ণসংখ্যা n, পূর্ণসংখ্যা m) {
    // assert m is a power of 2
    GGML_ASSERT((m & (m - 1)) == 0);
    ফেরত (n + m - 1) & ~(m - 1);
}

// TODO: move to ggml.h? (won't be able to inline)
স্থির bool ggml_are_same_layout(ধ্রুবক কাঠামো ggml_tensor * a, ধ্রুবক কাঠামো ggml_tensor * b) {
    যদি (a->type != b->type) {
        ফেরত false;
    }
    জন্য (পূর্ণসংখ্যা i = 0; i < GGML_MAX_DIMS; i++) {
        যদি (a->ne[i] != b->ne[i]) {
            ফেরত false;
        }
        যদি (a->nb[i] != b->nb[i]) {
            ফেরত false;
        }
    }
    ফেরত true;
}

স্থির bool ggml_op_is_empty(গণনা ggml_op op) {
    নির্বাচন (op) {
        ক্ষেত্র GGML_OP_NONE:
        ক্ষেত্র GGML_OP_RESHAPE:
        ক্ষেত্র GGML_OP_TRANSPOSE:
        ক্ষেত্র GGML_OP_VIEW:
        ক্ষেত্র GGML_OP_PERMUTE:
            ফেরত true;
        পূর্বনির্ধারিত:
            ফেরত false;
    }
}

স্থির অন্তঃরেখ bool ggml_impl_is_view(ধ্রুবক কাঠামো ggml_tensor * t) {
    ফেরত t->view_src != ফাঁকা;
}

স্থির অন্তঃরেখ দশমিক ggml_compute_softplus_f32(দশমিক প্রবেশ) {
    ফেরত (প্রবেশ > 20.0f) ? প্রবেশ : logf(1 + expf(প্রবেশ));
}
//
// logging
//

GGML_ATTRIBUTE_FORMAT(2, 3)
GGML_API শূন্য ggml_log_internal        (গণনা ggml_log_level level, ধ্রুবক অক্ষর * format, ...);
GGML_API শূন্য ggml_log_callback_default(গণনা ggml_log_level level, ধ্রুবক অক্ষর * text, শূন্য * user_তথ্য);

#define GGML_LOG(...)       ggml_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define GGML_LOG_INFO(...)  ggml_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define GGML_LOG_WARN(...)  ggml_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define GGML_LOG_ERROR(...) ggml_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GGML_LOG_DEBUG(...) ggml_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define GGML_LOG_CONT(...)  ggml_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

#define GGML_DEBUG 0

#যদি (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#অন্যথা
#define GGML_PRINT_DEBUG(...)
#endif

#যদি (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) GGML_LOG_DEBUG(__VA_ARGS__)
#অন্যথা
#define GGML_PRINT_DEBUG_5(...)
#endif

#যদি (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) GGML_LOG_DEBUG(__VA_ARGS__)
#অন্যথা
#define GGML_PRINT_DEBUG_10(...)
#endif

// tensor params

স্থির শূন্য ggml_set_op_params(কাঠামো ggml_tensor * tensor, ধ্রুবক শূন্য * params, size_t params_আকার) {
    GGML_ASSERT(tensor != ফাঁকা); // silence -Warray-bounds warnings
    প্রতিজ্ঞা(params_আকার <= GGML_MAX_OP_PARAMS);
    memcpy(tensor->op_params, params, params_আকার);
}

স্থির int32_t ggml_get_op_params_i32(ধ্রুবক কাঠামো ggml_tensor * tensor, uint32_t i) {
    প্রতিজ্ঞা(i < GGML_MAX_OP_PARAMS / আকার(int32_t));
    ফেরত ((ধ্রুবক int32_t *)(tensor->op_params))[i];
}

স্থির দশমিক ggml_get_op_params_f32(ধ্রুবক কাঠামো ggml_tensor * tensor, uint32_t i) {
    প্রতিজ্ঞা(i < GGML_MAX_OP_PARAMS / আকার(দশমিক));
    ফেরত ((ধ্রুবক দশমিক *)(tensor->op_params))[i];
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
স্থির শূন্য ggml_set_op_params_i32(কাঠামো ggml_tensor * tensor, uint32_t i, int32_t মান) {
    প্রতিজ্ঞা(i < GGML_MAX_OP_PARAMS / আকার(int32_t));
    ((int32_t *)(tensor->op_params))[i] = মান;
}

স্থির শূন্য ggml_set_op_params_f32(কাঠামো ggml_tensor * tensor, uint32_t i, দশমিক মান) {
    প্রতিজ্ঞা(i < GGML_MAX_OP_PARAMS / আকার(দশমিক));
    ((দশমিক *)(tensor->op_params))[i] = মান;
}

কাঠামো ggml_map_custom1_op_params {
    ggml_custom1_op_t  fun;
    পূর্ণসংখ্যা                n_tasks;
    শূন্য             * userdata;
};

কাঠামো ggml_map_custom2_op_params {
    ggml_custom2_op_t   fun;
    পূর্ণসংখ্যা                 n_tasks;
    শূন্য              * userdata;
};

কাঠামো ggml_map_custom3_op_params {
    ggml_custom3_op_t fun;
    পূর্ণসংখ্যা               n_tasks;
    শূন্য            * userdata;
};

কাঠামো ggml_custom_op_params {
    ggml_custom_op_t fun;
    পূর্ণসংখ্যা              n_tasks;
    শূন্য           * userdata;
};

// bitset

প্রকারলেখ uint32_t ggml_bitset_t;

static_assert(আকার(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#define BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#define BITSET_MASK (আকার(ggml_bitset_t)*8 - 1)

স্থির size_t ggml_bitset_আকার(size_t n) {
    ফেরত (n + BITSET_MASK) >> BITSET_SHR;
}

স্থির অন্তঃরেখ bool ggml_bitset_get(ধ্রুবক ggml_bitset_t * bitset, size_t i) {
    ফেরত !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
}

স্থির অন্তঃরেখ শূন্য ggml_bitset_set(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_MASK));
}

স্থির অন্তঃরেখ শূন্য ggml_bitset_clear(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_MASK));
}

// hash set

#define GGML_HASHSET_FULL ((size_t)-1)
#define GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)

কাঠামো ggml_hash_set {
    size_t আকার_2;
    ggml_bitset_t * used;       // whether or not the keys are in use i.e. set
    কাঠামো ggml_tensor ** চাবিগুলি; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};

কাঠামো ggml_hash_set ggml_hash_set_new(size_t আকার_2);
শূন্য                 ggml_hash_set_free(কাঠামো ggml_hash_set * hash_set);

// returns the minimum size for a hash set that can hold min_sz elements
size_t ggml_hash_আকার(size_t min_sz);

// remove all elements from the hash set
শূন্য ggml_hash_set_reset(কাঠামো ggml_hash_set * hash_set);

// returns true if key is in the hash set
স্থির bool ggml_hash_contains(ধ্রুবক কাঠামো ggml_hash_set * hash_set, কাঠামো ggml_tensor * চাবি);

// returns GGML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
স্থির size_t ggml_hash_find(ধ্রুবক কাঠামো ggml_hash_set * hash_set, ধ্রুবক কাঠামো ggml_tensor * চাবি);

// returns GGML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
স্থির size_t ggml_hash_ঢোকাও(কাঠামো ggml_hash_set * hash_set, কাঠামো ggml_tensor * চাবি);

// return index, asserts if table is full
স্থির size_t ggml_hash_find_or_ঢোকাও(কাঠামো ggml_hash_set * hash_set, কাঠামো ggml_tensor * চাবি);

// hash function for ggml_tensor
স্থির অন্তঃরেখ size_t ggml_hash(ধ্রুবক কাঠামো ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    ফেরত (size_t)(uintptr_t)p >> 4;
}

স্থির size_t ggml_hash_find(ধ্রুবক কাঠামো ggml_hash_set * hash_set, ধ্রুবক কাঠামো ggml_tensor * চাবি) {
    size_t h = ggml_hash(চাবি) % hash_set->আকার_2;

    // linear probing
    size_t i = h;
    যখন (ggml_bitset_get(hash_set->used, i) && hash_set->চাবিগুলি[i] != চাবি) {
        i = (i + 1) % hash_set->আকার_2;
        যদি (i == h) {
            // visited all hash table entries -> not found
            ফেরত GGML_HASHSET_FULL;
        }
    }
    ফেরত i;
}

স্থির bool ggml_hash_contains(ধ্রুবক কাঠামো ggml_hash_set * hash_set, কাঠামো ggml_tensor * চাবি) {
    size_t i = ggml_hash_find(hash_set, চাবি);
    ফেরত i != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, i);
}

স্থির size_t ggml_hash_ঢোকাও(কাঠামো ggml_hash_set * hash_set, কাঠামো ggml_tensor * চাবি) {
    size_t h = ggml_hash(চাবি) % hash_set->আকার_2;

    // linear probing
    size_t i = h;
    করো {
        যদি (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->চাবিগুলি[i] = চাবি;
            ফেরত i;
        }
        যদি (hash_set->চাবিগুলি[i] == চাবি) {
            ফেরত GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_set->আকার_2;
    } যখন (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

স্থির size_t ggml_hash_find_or_ঢোকাও(কাঠামো ggml_hash_set * hash_set, কাঠামো ggml_tensor * চাবি) {
    size_t h = ggml_hash(চাবি) % hash_set->আকার_2;

    // linear probing
    size_t i = h;
    করো {
        যদি (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->চাবিগুলি[i] = চাবি;
            ফেরত i;
        }
        যদি (hash_set->চাবিগুলি[i] == চাবি) {
            ফেরত i;
        }
        i = (i + 1) % hash_set->আকার_2;
    } যখন (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

// computation graph

গণনা ggml_cgraph_eval_order {
    GGML_CGRAPH_EVAL_ORDER_বাম_TO_ডান = 0,
    GGML_CGRAPH_EVAL_ORDER_ডান_TO_বাম,
    GGML_CGRAPH_EVAL_ORDER_গুনতি
};

কাঠামো ggml_cgraph {
    পূর্ণসংখ্যা আকার_2;    // maximum number of nodes/leafs/grads/grad_accs
    পূর্ণসংখ্যা n_nodes; // number of nodes currently in use
    পূর্ণসংখ্যা n_leafs; // number of leafs currently in use

    কাঠামো ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
    কাঠামো ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
    কাঠামো ggml_tensor ** grad_accs; // accumulators for node gradients
    কাঠামো ggml_tensor ** leafs;     // tensors with constant data
    int32_t             * use_counts;// number of uses of each tensor, indexed by hash table slot

    কাঠামো ggml_hash_set visited_hash_set;

    গণনা ggml_cgraph_eval_order order;

    // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
    // a value of 0 means it is not set and should be ignored
    uint64_t uid;
};

// returns a slice of cgraph with nodes [i0, i1)
// the slice does not have leafs or gradients
// if you need the gradients, get them from the original graph
কাঠামো ggml_cgraph ggml_graph_view(কাঠামো ggml_cgraph * cgraph, পূর্ণসংখ্যা i0, পূর্ণসংখ্যা i1);

// ggml-alloc.c: true if the operation can reuse memory from its sources
GGML_API bool ggml_op_can_inplace(গণনা ggml_op op);


// Memory allocation

GGML_API শূন্য * ggml_aligned_malloc(size_t আকার_2);
GGML_API শূন্য ggml_aligned_free(শূন্য * ptr, size_t আকার_2);

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

স্থির অন্তঃরেখ দশমিক fp32_from_bits(uint32_t w) {
    সংঘ {
        uint32_t as_bits;
        দশমিক as_মান;
    } fp32;
    fp32.as_bits = w;
    ফেরত fp32.as_মান;
}

স্থির অন্তঃরেখ uint32_t fp32_to_bits(দশমিক f) {
    সংঘ {
        দশমিক as_মান;
        uint32_t as_bits;
    } fp32;
    fp32.as_মান = f;
    ফেরত fp32.as_bits;
}

স্থির অন্তঃরেখ দশমিক ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    ধ্রুবক uint32_t w = (uint32_t) h << 16;
    ধ্রুবক uint32_t sign = w & UINT32_C(0x80000000);
    ধ্রুবক uint32_t two_w = w + w;

    ধ্রুবক uint32_t exp_offset = UINT32_C(0xE0) << 23;
#যদি (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    ধ্রুবক দশমিক exp_scale = 0x1.0p-112f;
#অন্যথা
    ধ্রুবক দশমিক exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    ধ্রুবক দশমিক normalized_মান = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    ধ্রুবক uint32_t magic_mask = UINT32_C(126) << 23;
    ধ্রুবক দশমিক magic_bias = 0.5f;
    ধ্রুবক দশমিক denormalized_মান = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    ধ্রুবক uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    ধ্রুবক uint32_t ফলাফল = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_মান) : fp32_to_bits(normalized_মান));
    ফেরত fp32_from_bits(ফলাফল);
}

স্থির অন্তঃরেখ ggml_fp16_t ggml_compute_fp32_to_fp16(দশমিক f) {
#যদি (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    ধ্রুবক দশমিক scale_to_inf = 0x1.0p+112f;
    ধ্রুবক দশমিক scale_to_zero = 0x1.0p-110f;
#অন্যথা
    ধ্রুবক দশমিক scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    ধ্রুবক দশমিক scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    দশমিক base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    ধ্রুবক uint32_t w = fp32_to_bits(f);
    ধ্রুবক uint32_t shl1_w = w + w;
    ধ্রুবক uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    যদি (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    ধ্রুবক uint32_t bits = fp32_to_bits(base);
    ধ্রুবক uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    ধ্রুবক uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    ধ্রুবক uint32_t nonsign = exp_bits + mantissa_bits;
    ফেরত (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

স্থির অন্তঃরেখ দশমিক ggml_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;  // Stores the raw bit representation of the float

    // Handle special case for minimum exponent (denormalized float)
    যদি (x == 0) {
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
    অন্যথা {
        // Construct normalized float by shifting exponent into position:
        // - Exponent field: 8 bits (positions 30-23)
        // - Mantissa: 0 (implicit leading 1)
        // Value = 2^(x - 127)
        bits = (uint32_t) x << 23;
    }

    দশমিক ফলাফল;  // Final float value
                   // Safely reinterpret bit pattern as float without type-punning issues
    memcpy(&ফলাফল, &bits, আকার(দশমিক));
    ফেরত ফলাফল;
}

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
স্থির অন্তঃরেখ দশমিক ggml_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;

    // For x < 2: use precomputed denormal patterns
    যদি (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        bits = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    অন্যথা {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        bits = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    দশমিক ফলাফল;
    memcpy(&ফলাফল, &bits, আকার(দশমিক));
    ফেরত ফলাফল;
}

#define GGML_E8M0_TO_FP32(x) ggml_e8m0_to_fp32(x)
#define GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
স্থির অন্তঃরেখ দশমিক ggml_ue4m3_to_fp32(uint8_t x) {
    যদি (x == 0 || x == 0x7F) {
        ফেরত 0.0f;
    }
    পূর্ণসংখ্যা   exp = (x >> 3) & 0xF;
    পূর্ণসংখ্যা   man = x & 0x7;
    দশমিক raw;
    যদি (exp == 0) {
        raw = ldexpf((দশমিক) man, -9);
    } অন্যথা {
        raw = ldexpf(1.0f + (দশমিক) man / 8.0f, exp - 7);
    }
    ফেরত raw * 0.5f;
}

স্থির অন্তঃরেখ uint8_t ggml_fp32_to_ue4m3(দশমিক x) {
    যদি (!(x > 0.0f)) {
        ফেরত 0;
    }
    যদি (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t bits;
    memcpy(&bits, &x, 4);
    পূর্ণসংখ্যা fp32_exp  = ((bits >> 23) & 0xFF) - 127;
    পূর্ণসংখ্যা fp32_man  = (bits >> 20) & 0x7;
    পূর্ণসংখ্যা ue4m3_exp = fp32_exp + 7;
    যদি (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        পূর্ণসংখ্যা man = (পূর্ণসংখ্যা) (x * 512.0f + 0.5f);
        যদি (man > 7) {
            man = 7;
        }
        যদি (man < 1) {
            ফেরত 0;
        }
        ফেরত (uint8_t) man;
    }
    যদি (ue4m3_exp >= 15) {
        ফেরত 0x7E;
    }
    পূর্ণসংখ্যা round_bit = (bits >> 19) & 1;
    পূর্ণসংখ্যা ue4m3_man = fp32_man + round_bit;
    যদি (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        যদি (ue4m3_exp >= 15) {
            ফেরত 0x7E;
        }
    }
    ফেরত (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
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
স্থির অন্তঃরেখ দশমিক ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    সংঘ {
        দশমিক f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    ফেরত u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
স্থির অন্তঃরেখ ggml_bf16_t ggml_compute_fp32_to_bf16(দশমিক s) {
    ggml_bf16_t h;
    সংঘ {
        দশমিক f;
        uint32_t i;
    } u;
    u.f = s;
    যদি ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.bits = (u.i >> 16) | 64; /* force to quiet */
        ফেরত h;
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    ফেরত h;
}

#define GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)
#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)

স্থির অন্তঃরেখ int32_t ggml_গ্রন্থি_get_use_গুনতি(ধ্রুবক কাঠামো ggml_cgraph * cgraph, পূর্ণসংখ্যা গ্রন্থি_idx) {
    ধ্রুবক কাঠামো ggml_tensor * গ্রন্থি = cgraph->nodes[গ্রন্থি_idx];

    size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, গ্রন্থি);
    যদি (!ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        ফেরত 0;
    }
    ফেরত cgraph->use_counts[hash_pos];
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
স্থির অন্তঃরেখ bool ggml_গ্রন্থি_has_n_uses(ধ্রুবক কাঠামো ggml_cgraph * cgraph, পূর্ণসংখ্যা গ্রন্থি_idx, int32_t n_uses) {
    ধ্রুবক কাঠামো ggml_tensor * গ্রন্থি = cgraph->nodes[গ্রন্থি_idx];

    // check the use count against how many we're replacing
    যদি (ggml_গ্রন্থি_get_use_গুনতি(cgraph, গ্রন্থি_idx) != n_uses) {
        ফেরত false;
    }

    // if node is a view, some other node might be using the intermediate result
    // via the view source.
    যদি (গ্রন্থি->view_src) {
        ফেরত false;
    }

    // If the user requested output for the node, can't fuse
    যদি (গ্রন্থি->flags & GGML_TENSOR_FLAG_OUTPUT) {
        ফেরত false;
    }

    ফেরত true;
}

// Returns true if nodes with indices { node_idxs } are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
স্থির অন্তঃরেখ bool ggml_can_fuse_ext(ধ্রুবক কাঠামো ggml_cgraph * cgraph, ধ্রুবক পূর্ণসংখ্যা * গ্রন্থি_idxs, ধ্রুবক গণনা ggml_op * ops, পূর্ণসংখ্যা অঙ্ক_ops) {
    জন্য (পূর্ণসংখ্যা i = 0; i < অঙ্ক_ops; ++i) {
        যদি (গ্রন্থি_idxs[i] >= cgraph->n_nodes) {
            ফেরত false;
        }

        কাঠামো ggml_tensor * গ্রন্থি = cgraph->nodes[গ্রন্থি_idxs[i]];
        যদি (গ্রন্থি->op != ops[i]) {
            ফেরত false;
        }
        যদি ((গ্রন্থি->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ফেরত false;
        }
        যদি (i < অঙ্ক_ops - 1 && !ggml_গ্রন্থি_has_n_uses(cgraph, গ্রন্থি_idxs[i], 1)) {
            ফেরত false;
        }
        যদি (i > 0) {
            কাঠামো ggml_tensor * পূর্ব = cgraph->nodes[গ্রন্থি_idxs[i - 1]];
            যদি (গ্রন্থি->src[0] != পূর্ব && গ্রন্থি->src[1] != পূর্ব) {
                ফেরত false;
            }
            যদি (!ggml_are_same_shape(গ্রন্থি, পূর্ব)) {
                ফেরত false;
            }
        }
    }
    ফেরত true;
}

// same as above, for sequential indices starting at node_idx
স্থির অন্তঃরেখ bool ggml_can_fuse(ধ্রুবক কাঠামো ggml_cgraph * cgraph, পূর্ণসংখ্যা গ্রন্থি_idx, ধ্রুবক গণনা ggml_op * ops, পূর্ণসংখ্যা অঙ্ক_ops) {
    প্রতিজ্ঞা(অঙ্ক_ops < 32);

    যদি (গ্রন্থি_idx + অঙ্ক_ops > cgraph->n_nodes) {
        ফেরত false;
    }

    পূর্ণসংখ্যা idxs[32];
    জন্য (পূর্ণসংখ্যা i = 0; i < অঙ্ক_ops; ++i) {
        idxs[i] = গ্রন্থি_idx + i;
    }

    ফেরত ggml_can_fuse_ext(cgraph, idxs, ops, অঙ্ক_ops);
}

GGML_API bool ggml_can_fuse_subgraph_ext(ধ্রুবক কাঠামো ggml_cgraph * cgraph,
                                         ধ্রুবক পূর্ণসংখ্যা *                গ্রন্থি_idxs,
                                         পূর্ণসংখ্যা                        গুনতি,
                                         ধ্রুবক গণনা ggml_op *       ops,
                                         ধ্রুবক পূর্ণসংখ্যা *                outputs,
                                         পূর্ণসংখ্যা                        অঙ্ক_outputs);

// Returns true if the subgraph formed by {node_idxs} can be fused
// checks whethers all nodes which are not part of outputs can be elided
// by checking if their num_uses are confined to the subgraph
স্থির অন্তঃরেখ bool ggml_can_fuse_subgraph(ধ্রুবক কাঠামো ggml_cgraph * cgraph,
                                          পূর্ণসংখ্যা                        গ্রন্থি_idx,
                                          পূর্ণসংখ্যা                        গুনতি,
                                          ধ্রুবক গণনা ggml_op *       ops,
                                          ধ্রুবক পূর্ণসংখ্যা *                outputs,
                                          পূর্ণসংখ্যা                        অঙ্ক_outputs) {
    GGML_ASSERT(গুনতি < 32);
    যদি (গ্রন্থি_idx + গুনতি > cgraph->n_nodes) {
        ফেরত false;
    }

    পূর্ণসংখ্যা idxs[32];

    জন্য (পূর্ণসংখ্যা i = 0; i < গুনতি; ++i) {
        idxs[i] = গ্রন্থি_idx + i;
    }

    ফেরত ggml_can_fuse_subgraph_ext(cgraph, idxs, গুনতি, ops, outputs, অঙ্ক_outputs);
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#অন্তর্ভাব <array>
#অন্তর্ভাব <initializer_list>
#অন্তর্ভাব <vector>

// nicer C++ syntax for ggml_can_fuse
অন্তঃরেখ bool ggml_can_fuse(ধ্রুবক কাঠামো ggml_cgraph * cgraph, পূর্ণসংখ্যা গ্রন্থি_idx, std::initializer_list<গণনা ggml_op> ops) {
    ফেরত ggml_can_fuse(cgraph, গ্রন্থি_idx, ops.begin(), (পূর্ণসংখ্যা)ops.আকার_2());
}

অন্তঃরেখ bool ggml_can_fuse_subgraph(ধ্রুবক কাঠামো ggml_cgraph *          cgraph,
                                   পূর্ণসংখ্যা                                 start_idx,
                                   std::initializer_list<গণনা ggml_op> ops,
                                   std::initializer_list<পূর্ণসংখ্যা>          outputs = {}) {
    ফেরত ggml_can_fuse_subgraph(cgraph, start_idx, ops.আকার_2(), ops.begin(), outputs.begin(), outputs.আকার_2());
}

// Return true if the edges in the graph match expectations.
অন্তঃরেখ bool ggml_check_edges(ধ্রুবক কাঠামো ggml_cgraph *                cgraph,
                             পূর্ণসংখ্যা                                       start_idx,
                             std::initializer_list<std::array<পূর্ণসংখ্যা, 3>> edges) {
    জন্য (ধ্রুবক auto & edge : edges) {
        পূর্ণসংখ্যা dst_গ্রন্থি = edge[0];
        পূর্ণসংখ্যা src_idx  = edge[1];
        পূর্ণসংখ্যা src_গ্রন্থি = edge[2];
        যদি (cgraph->nodes[start_idx + dst_গ্রন্থি]->src[src_idx] != cgraph->nodes[start_idx + src_গ্রন্থি]) {
            ফেরত false;
        }
    }
    ফেরত true;
}

// expose GGUF internals for test code
GGML_API size_t gguf_type_আকার(গণনা gguf_type type);
GGML_API শূন্য gguf_write_to_buf(ধ্রুবক কাঠামো gguf_context * ctx, std::vector<int8_t> & buf, bool only_meta);
#endif // __cplusplus
