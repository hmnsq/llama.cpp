#pragma once

// GGML internal header

#ضمن "ggml.h"
#ضمن "gguf.h"

#ضمن <assert.h>
#ضمن <math.h>
#ضمن <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <string.h>

#ifdef __ARM_FEATURE_SVE
#ضمن <arm_sve.h>
#endif // __ARM_FEATURE_SVE

#إذا defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#ضمن <arm_neon.h>
#endif

#ifdef __cplusplus
خارجي "C" {
#endif

فراغ ggml_print_backtrace(فراغ);

uint64_t ggml_graph_next_uid(فراغ);

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
        #إذا defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
            #define static_assert(cond, رسالة) _Static_assert(cond, رسالة)
        #وإلا
            #define static_assert(cond, رسالة) بنية global_scope_noop_trick
        #endif
    #endif
#endif

ساكن مضمن صحيح ggml_up32(صحيح n) {
    إرجاع (n + 31) & ~31;
}

//static inline int ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

ساكن مضمن صحيح ggml_up(صحيح n, صحيح m) {
    // assert m is a power of 2
    GGML_ASSERT((m & (m - 1)) == 0);
    إرجاع (n + m - 1) & ~(m - 1);
}

// TODO: move to ggml.h? (won't be able to inline)
ساكن bool ggml_are_same_layout(ثابت بنية ggml_tensor * a, ثابت بنية ggml_tensor * b) {
    إذا (a->type != b->type) {
        إرجاع false;
    }
    لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
        إذا (a->ne[i] != b->ne[i]) {
            إرجاع false;
        }
        إذا (a->nb[i] != b->nb[i]) {
            إرجاع false;
        }
    }
    إرجاع true;
}

ساكن bool ggml_op_is_empty(تعداد ggml_op op) {
    بدل (op) {
        حالة GGML_OP_NONE:
        حالة GGML_OP_RESHAPE:
        حالة GGML_OP_TRANSPOSE:
        حالة GGML_OP_VIEW:
        حالة GGML_OP_PERMUTE:
            إرجاع true;
        افتراضي:
            إرجاع false;
    }
}

ساكن مضمن bool ggml_impl_is_view(ثابت بنية ggml_tensor * t) {
    إرجاع t->view_src != عدم;
}

ساكن مضمن عائم ggml_compute_softplus_f32(عائم إدخال) {
    إرجاع (إدخال > 20.0f) ? إدخال : logf(1 + expf(إدخال));
}
//
// logging
//

GGML_ATTRIBUTE_FORMAT(2, 3)
GGML_API فراغ ggml_log_internal        (تعداد ggml_log_level level, ثابت حرف * format, ...);
GGML_API فراغ ggml_log_callback_default(تعداد ggml_log_level level, ثابت حرف * text, فراغ * user_data);

#define GGML_LOG(...)       ggml_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define GGML_LOG_INFO(...)  ggml_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define GGML_LOG_WARN(...)  ggml_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define GGML_LOG_ERROR(...) ggml_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GGML_LOG_DEBUG(...) ggml_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define GGML_LOG_CONT(...)  ggml_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

#define GGML_DEBUG 0

#إذا (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#وإلا
#define GGML_PRINT_DEBUG(...)
#endif

#إذا (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) GGML_LOG_DEBUG(__VA_ARGS__)
#وإلا
#define GGML_PRINT_DEBUG_5(...)
#endif

#إذا (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) GGML_LOG_DEBUG(__VA_ARGS__)
#وإلا
#define GGML_PRINT_DEBUG_10(...)
#endif

// tensor params

ساكن فراغ ggml_set_op_params(بنية ggml_tensor * tensor, ثابت فراغ * params, size_t params_size) {
    GGML_ASSERT(tensor != عدم); // silence -Warray-bounds warnings
    أكد(params_size <= GGML_MAX_OP_PARAMS);
    memcpy(tensor->op_params, params, params_size);
}

ساكن int32_t ggml_get_op_params_i32(ثابت بنية ggml_tensor * tensor, uint32_t i) {
    أكد(i < GGML_MAX_OP_PARAMS / حجم(int32_t));
    إرجاع ((ثابت int32_t *)(tensor->op_params))[i];
}

ساكن عائم ggml_get_op_params_f32(ثابت بنية ggml_tensor * tensor, uint32_t i) {
    أكد(i < GGML_MAX_OP_PARAMS / حجم(عائم));
    إرجاع ((ثابت عائم *)(tensor->op_params))[i];
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
ساكن فراغ ggml_set_op_params_i32(بنية ggml_tensor * tensor, uint32_t i, int32_t قيمة) {
    أكد(i < GGML_MAX_OP_PARAMS / حجم(int32_t));
    ((int32_t *)(tensor->op_params))[i] = قيمة;
}

ساكن فراغ ggml_set_op_params_f32(بنية ggml_tensor * tensor, uint32_t i, عائم قيمة) {
    أكد(i < GGML_MAX_OP_PARAMS / حجم(عائم));
    ((عائم *)(tensor->op_params))[i] = قيمة;
}

بنية ggml_map_custom1_op_params {
    ggml_custom1_op_t  fun;
    صحيح                n_tasks;
    فراغ             * userdata;
};

بنية ggml_map_custom2_op_params {
    ggml_custom2_op_t   fun;
    صحيح                 n_tasks;
    فراغ              * userdata;
};

بنية ggml_map_custom3_op_params {
    ggml_custom3_op_t fun;
    صحيح               n_tasks;
    فراغ            * userdata;
};

بنية ggml_custom_op_params {
    ggml_custom_op_t fun;
    صحيح              n_tasks;
    فراغ           * userdata;
};

// bitset

تعريفنوع uint32_t ggml_bitset_t;

static_assert(حجم(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#define BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#define BITSET_MASK (حجم(ggml_bitset_t)*8 - 1)

ساكن size_t ggml_bitset_size(size_t n) {
    إرجاع (n + BITSET_MASK) >> BITSET_SHR;
}

ساكن مضمن bool ggml_bitset_get(ثابت ggml_bitset_t * bitset, size_t i) {
    إرجاع !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
}

ساكن مضمن فراغ ggml_bitset_set(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_MASK));
}

ساكن مضمن فراغ ggml_bitset_clear(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_MASK));
}

// hash set

#define GGML_HASHSET_FULL ((size_t)-1)
#define GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)

بنية ggml_hash_set {
    size_t size;
    ggml_bitset_t * used;       // whether or not the keys are in use i.e. set
    بنية ggml_tensor ** مفاتيح; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};

بنية ggml_hash_set ggml_hash_set_new(size_t size);
فراغ                 ggml_hash_set_free(بنية ggml_hash_set * hash_set);

// returns the minimum size for a hash set that can hold min_sz elements
size_t ggml_hash_size(size_t min_sz);

// remove all elements from the hash set
فراغ ggml_hash_set_reset(بنية ggml_hash_set * hash_set);

// returns true if key is in the hash set
ساكن bool ggml_hash_contains(ثابت بنية ggml_hash_set * hash_set, بنية ggml_tensor * مفتاح);

// returns GGML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
ساكن size_t ggml_hash_find(ثابت بنية ggml_hash_set * hash_set, ثابت بنية ggml_tensor * مفتاح);

// returns GGML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
ساكن size_t ggml_hash_أدرج(بنية ggml_hash_set * hash_set, بنية ggml_tensor * مفتاح);

// return index, asserts if table is full
ساكن size_t ggml_hash_find_or_أدرج(بنية ggml_hash_set * hash_set, بنية ggml_tensor * مفتاح);

// hash function for ggml_tensor
ساكن مضمن size_t ggml_hash(ثابت بنية ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    إرجاع (size_t)(uintptr_t)p >> 4;
}

ساكن size_t ggml_hash_find(ثابت بنية ggml_hash_set * hash_set, ثابت بنية ggml_tensor * مفتاح) {
    size_t h = ggml_hash(مفتاح) % hash_set->size;

    // linear probing
    size_t i = h;
    بينما (ggml_bitset_get(hash_set->used, i) && hash_set->مفاتيح[i] != مفتاح) {
        i = (i + 1) % hash_set->size;
        إذا (i == h) {
            // visited all hash table entries -> not found
            إرجاع GGML_HASHSET_FULL;
        }
    }
    إرجاع i;
}

ساكن bool ggml_hash_contains(ثابت بنية ggml_hash_set * hash_set, بنية ggml_tensor * مفتاح) {
    size_t i = ggml_hash_find(hash_set, مفتاح);
    إرجاع i != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, i);
}

ساكن size_t ggml_hash_أدرج(بنية ggml_hash_set * hash_set, بنية ggml_tensor * مفتاح) {
    size_t h = ggml_hash(مفتاح) % hash_set->size;

    // linear probing
    size_t i = h;
    افعل {
        إذا (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->مفاتيح[i] = مفتاح;
            إرجاع i;
        }
        إذا (hash_set->مفاتيح[i] == مفتاح) {
            إرجاع GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_set->size;
    } بينما (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

ساكن size_t ggml_hash_find_or_أدرج(بنية ggml_hash_set * hash_set, بنية ggml_tensor * مفتاح) {
    size_t h = ggml_hash(مفتاح) % hash_set->size;

    // linear probing
    size_t i = h;
    افعل {
        إذا (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->مفاتيح[i] = مفتاح;
            إرجاع i;
        }
        إذا (hash_set->مفاتيح[i] == مفتاح) {
            إرجاع i;
        }
        i = (i + 1) % hash_set->size;
    } بينما (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

// computation graph

تعداد ggml_cgraph_eval_order {
    GGML_CGRAPH_EVAL_ORDER_يسار_TO_يمين = 0,
    GGML_CGRAPH_EVAL_ORDER_يمين_TO_يسار,
    GGML_CGRAPH_EVAL_ORDER_تعداد
};

بنية ggml_cgraph {
    صحيح size;    // maximum number of nodes/leafs/grads/grad_accs
    صحيح n_nodes; // number of nodes currently in use
    صحيح n_leafs; // number of leafs currently in use

    بنية ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
    بنية ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
    بنية ggml_tensor ** grad_accs; // accumulators for node gradients
    بنية ggml_tensor ** leafs;     // tensors with constant data
    int32_t             * use_counts;// number of uses of each tensor, indexed by hash table slot

    بنية ggml_hash_set visited_hash_set;

    تعداد ggml_cgraph_eval_order order;

    // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
    // a value of 0 means it is not set and should be ignored
    uint64_t uid;
};

// returns a slice of cgraph with nodes [i0, i1)
// the slice does not have leafs or gradients
// if you need the gradients, get them from the original graph
بنية ggml_cgraph ggml_graph_view(بنية ggml_cgraph * cgraph, صحيح i0, صحيح i1);

// ggml-alloc.c: true if the operation can reuse memory from its sources
GGML_API bool ggml_op_can_inplace(تعداد ggml_op op);


// Memory allocation

GGML_API فراغ * ggml_aligned_malloc(size_t size);
GGML_API فراغ ggml_aligned_free(فراغ * ptr, size_t size);

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

ساكن مضمن عائم fp32_from_bits(uint32_t w) {
    اتحاد {
        uint32_t as_bits;
        عائم as_قيمة;
    } fp32;
    fp32.as_bits = w;
    إرجاع fp32.as_قيمة;
}

ساكن مضمن uint32_t fp32_to_bits(عائم f) {
    اتحاد {
        عائم as_قيمة;
        uint32_t as_bits;
    } fp32;
    fp32.as_قيمة = f;
    إرجاع fp32.as_bits;
}

ساكن مضمن عائم ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    ثابت uint32_t w = (uint32_t) h << 16;
    ثابت uint32_t sign = w & UINT32_C(0x80000000);
    ثابت uint32_t two_w = w + w;

    ثابت uint32_t exp_offset = UINT32_C(0xE0) << 23;
#إذا (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    ثابت عائم exp_scale = 0x1.0p-112f;
#وإلا
    ثابت عائم exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    ثابت عائم normalized_قيمة = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    ثابت uint32_t magic_mask = UINT32_C(126) << 23;
    ثابت عائم magic_bias = 0.5f;
    ثابت عائم denormalized_قيمة = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    ثابت uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    ثابت uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_قيمة) : fp32_to_bits(normalized_قيمة));
    إرجاع fp32_from_bits(result);
}

ساكن مضمن ggml_fp16_t ggml_compute_fp32_to_fp16(عائم f) {
#إذا (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    ثابت عائم scale_to_inf = 0x1.0p+112f;
    ثابت عائم scale_to_zero = 0x1.0p-110f;
#وإلا
    ثابت عائم scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    ثابت عائم scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    عائم base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    ثابت uint32_t w = fp32_to_bits(f);
    ثابت uint32_t shl1_w = w + w;
    ثابت uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    إذا (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    ثابت uint32_t bits = fp32_to_bits(base);
    ثابت uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    ثابت uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    ثابت uint32_t nonsign = exp_bits + mantissa_bits;
    إرجاع (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

ساكن مضمن عائم ggml_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;  // Stores the raw bit representation of the float

    // Handle special case for minimum exponent (denormalized float)
    إذا (x == 0) {
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
    وإلا {
        // Construct normalized float by shifting exponent into position:
        // - Exponent field: 8 bits (positions 30-23)
        // - Mantissa: 0 (implicit leading 1)
        // Value = 2^(x - 127)
        bits = (uint32_t) x << 23;
    }

    عائم result;  // Final float value
                   // Safely reinterpret bit pattern as float without type-punning issues
    memcpy(&result, &bits, حجم(عائم));
    إرجاع result;
}

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
ساكن مضمن عائم ggml_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;

    // For x < 2: use precomputed denormal patterns
    إذا (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        bits = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    وإلا {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        bits = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    عائم result;
    memcpy(&result, &bits, حجم(عائم));
    إرجاع result;
}

#define GGML_E8M0_TO_FP32(x) ggml_e8m0_to_fp32(x)
#define GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
ساكن مضمن عائم ggml_ue4m3_to_fp32(uint8_t x) {
    إذا (x == 0 || x == 0x7F) {
        إرجاع 0.0f;
    }
    صحيح   exp = (x >> 3) & 0xF;
    صحيح   man = x & 0x7;
    عائم raw;
    إذا (exp == 0) {
        raw = ldexpf((عائم) man, -9);
    } وإلا {
        raw = ldexpf(1.0f + (عائم) man / 8.0f, exp - 7);
    }
    إرجاع raw * 0.5f;
}

ساكن مضمن uint8_t ggml_fp32_to_ue4m3(عائم x) {
    إذا (!(x > 0.0f)) {
        إرجاع 0;
    }
    إذا (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t bits;
    memcpy(&bits, &x, 4);
    صحيح fp32_exp  = ((bits >> 23) & 0xFF) - 127;
    صحيح fp32_man  = (bits >> 20) & 0x7;
    صحيح ue4m3_exp = fp32_exp + 7;
    إذا (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        صحيح man = (صحيح) (x * 512.0f + 0.5f);
        إذا (man > 7) {
            man = 7;
        }
        إذا (man < 1) {
            إرجاع 0;
        }
        إرجاع (uint8_t) man;
    }
    إذا (ue4m3_exp >= 15) {
        إرجاع 0x7E;
    }
    صحيح round_bit = (bits >> 19) & 1;
    صحيح ue4m3_man = fp32_man + round_bit;
    إذا (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        إذا (ue4m3_exp >= 15) {
            إرجاع 0x7E;
        }
    }
    إرجاع (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
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
ساكن مضمن عائم ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    اتحاد {
        عائم f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    إرجاع u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
ساكن مضمن ggml_bf16_t ggml_compute_fp32_to_bf16(عائم s) {
    ggml_bf16_t h;
    اتحاد {
        عائم f;
        uint32_t i;
    } u;
    u.f = s;
    إذا ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.bits = (u.i >> 16) | 64; /* force to quiet */
        إرجاع h;
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    إرجاع h;
}

#define GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)
#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)

ساكن مضمن int32_t ggml_عقدة_get_use_تعداد(ثابت بنية ggml_cgraph * cgraph, صحيح عقدة_idx) {
    ثابت بنية ggml_tensor * عقدة = cgraph->nodes[عقدة_idx];

    size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, عقدة);
    إذا (!ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        إرجاع 0;
    }
    إرجاع cgraph->use_counts[hash_pos];
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
ساكن مضمن bool ggml_عقدة_has_n_uses(ثابت بنية ggml_cgraph * cgraph, صحيح عقدة_idx, int32_t n_uses) {
    ثابت بنية ggml_tensor * عقدة = cgraph->nodes[عقدة_idx];

    // check the use count against how many we're replacing
    إذا (ggml_عقدة_get_use_تعداد(cgraph, عقدة_idx) != n_uses) {
        إرجاع false;
    }

    // if node is a view, some other node might be using the intermediate result
    // via the view source.
    إذا (عقدة->view_src) {
        إرجاع false;
    }

    // If the user requested output for the node, can't fuse
    إذا (عقدة->flags & GGML_TENSOR_FLAG_OUTPUT) {
        إرجاع false;
    }

    إرجاع true;
}

// Returns true if nodes with indices { node_idxs } are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
ساكن مضمن bool ggml_can_fuse_ext(ثابت بنية ggml_cgraph * cgraph, ثابت صحيح * عقدة_idxs, ثابت تعداد ggml_op * ops, صحيح رقم_ops) {
    لكل (صحيح i = 0; i < رقم_ops; ++i) {
        إذا (عقدة_idxs[i] >= cgraph->n_nodes) {
            إرجاع false;
        }

        بنية ggml_tensor * عقدة = cgraph->nodes[عقدة_idxs[i]];
        إذا (عقدة->op != ops[i]) {
            إرجاع false;
        }
        إذا ((عقدة->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            إرجاع false;
        }
        إذا (i < رقم_ops - 1 && !ggml_عقدة_has_n_uses(cgraph, عقدة_idxs[i], 1)) {
            إرجاع false;
        }
        إذا (i > 0) {
            بنية ggml_tensor * prev = cgraph->nodes[عقدة_idxs[i - 1]];
            إذا (عقدة->src[0] != prev && عقدة->src[1] != prev) {
                إرجاع false;
            }
            إذا (!ggml_are_same_shape(عقدة, prev)) {
                إرجاع false;
            }
        }
    }
    إرجاع true;
}

// same as above, for sequential indices starting at node_idx
ساكن مضمن bool ggml_can_fuse(ثابت بنية ggml_cgraph * cgraph, صحيح عقدة_idx, ثابت تعداد ggml_op * ops, صحيح رقم_ops) {
    أكد(رقم_ops < 32);

    إذا (عقدة_idx + رقم_ops > cgraph->n_nodes) {
        إرجاع false;
    }

    صحيح idxs[32];
    لكل (صحيح i = 0; i < رقم_ops; ++i) {
        idxs[i] = عقدة_idx + i;
    }

    إرجاع ggml_can_fuse_ext(cgraph, idxs, ops, رقم_ops);
}

GGML_API bool ggml_can_fuse_subgraph_ext(ثابت بنية ggml_cgraph * cgraph,
                                         ثابت صحيح *                عقدة_idxs,
                                         صحيح                        تعداد_2,
                                         ثابت تعداد ggml_op *       ops,
                                         ثابت صحيح *                outputs,
                                         صحيح                        رقم_outputs);

// Returns true if the subgraph formed by {node_idxs} can be fused
// checks whethers all nodes which are not part of outputs can be elided
// by checking if their num_uses are confined to the subgraph
ساكن مضمن bool ggml_can_fuse_subgraph(ثابت بنية ggml_cgraph * cgraph,
                                          صحيح                        عقدة_idx,
                                          صحيح                        تعداد_2,
                                          ثابت تعداد ggml_op *       ops,
                                          ثابت صحيح *                outputs,
                                          صحيح                        رقم_outputs) {
    GGML_ASSERT(تعداد_2 < 32);
    إذا (عقدة_idx + تعداد_2 > cgraph->n_nodes) {
        إرجاع false;
    }

    صحيح idxs[32];

    لكل (صحيح i = 0; i < تعداد_2; ++i) {
        idxs[i] = عقدة_idx + i;
    }

    إرجاع ggml_can_fuse_subgraph_ext(cgraph, idxs, تعداد_2, ops, outputs, رقم_outputs);
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#ضمن <array>
#ضمن <initializer_list>
#ضمن <vector>

// nicer C++ syntax for ggml_can_fuse
مضمن bool ggml_can_fuse(ثابت بنية ggml_cgraph * cgraph, صحيح عقدة_idx, std::initializer_list<تعداد ggml_op> ops) {
    إرجاع ggml_can_fuse(cgraph, عقدة_idx, ops.begin(), (صحيح)ops.size());
}

مضمن bool ggml_can_fuse_subgraph(ثابت بنية ggml_cgraph *          cgraph,
                                   صحيح                                 start_idx,
                                   std::initializer_list<تعداد ggml_op> ops,
                                   std::initializer_list<صحيح>          outputs = {}) {
    إرجاع ggml_can_fuse_subgraph(cgraph, start_idx, ops.size(), ops.begin(), outputs.begin(), outputs.size());
}

// Return true if the edges in the graph match expectations.
مضمن bool ggml_check_edges(ثابت بنية ggml_cgraph *                cgraph,
                             صحيح                                       start_idx,
                             std::initializer_list<std::array<صحيح, 3>> edges) {
    لكل (ثابت auto & edge : edges) {
        صحيح dst_عقدة = edge[0];
        صحيح src_idx  = edge[1];
        صحيح src_عقدة = edge[2];
        إذا (cgraph->nodes[start_idx + dst_عقدة]->src[src_idx] != cgraph->nodes[start_idx + src_عقدة]) {
            إرجاع false;
        }
    }
    إرجاع true;
}

// expose GGUF internals for test code
GGML_API size_t gguf_type_size(تعداد gguf_type type);
GGML_API فراغ gguf_write_to_buf(ثابت بنية gguf_context * ctx, std::vector<int8_t> & buf, bool only_meta);
#endif // __cplusplus
