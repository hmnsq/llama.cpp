#pragma once

// GGML internal header

#شامل "ggml.h"
#شامل "gguf.h"

#شامل <assert.h>
#شامل <math.h>
#شامل <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#شامل <stdbool.h>
#شامل <stdint.h>
#شامل <string.h>

#ifdef __ARM_FEATURE_SVE
#شامل <arm_sve.h>
#endif // __ARM_FEATURE_SVE

#اگر defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#شامل <arm_neon.h>
#endif

#ifdef __cplusplus
بیرونی "C" {
#endif

تهی ggml_print_backtrace(تهی);

uint64_t ggml_graph_next_uid(تهی);

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
        #اگر defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
            #define static_assert(cond, پیام) _Static_assert(cond, پیام)
        #وگرنه
            #define static_assert(cond, پیام) ساختار global_scope_noop_trick
        #endif
    #endif
#endif

ایستا درونخطی صحیح ggml_up32(صحیح n) {
    بازگشت (n + 31) & ~31;
}

//static inline int ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

ایستا درونخطی صحیح ggml_up(صحیح n, صحیح m) {
    // assert m is a power of 2
    GGML_ASSERT((m & (m - 1)) == 0);
    بازگشت (n + m - 1) & ~(m - 1);
}

// TODO: move to ggml.h? (won't be able to inline)
ایستا bool ggml_are_same_layout(ثابت ساختار ggml_tensor * a, ثابت ساختار ggml_tensor * b) {
    اگر (a->type != b->type) {
        بازگشت false;
    }
    برای (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
        اگر (a->ne[i] != b->ne[i]) {
            بازگشت false;
        }
        اگر (a->nb[i] != b->nb[i]) {
            بازگشت false;
        }
    }
    بازگشت true;
}

ایستا bool ggml_op_is_empty(شمارش ggml_op op) {
    انتخاب (op) {
        حالت GGML_OP_NONE:
        حالت GGML_OP_RESHAPE:
        حالت GGML_OP_TRANSPOSE:
        حالت GGML_OP_VIEW:
        حالت GGML_OP_PERMUTE:
            بازگشت true;
        پیشفرض:
            بازگشت false;
    }
}

ایستا درونخطی bool ggml_impl_is_view(ثابت ساختار ggml_tensor * t) {
    بازگشت t->view_src != پوچ;
}

ایستا درونخطی اعشاری ggml_compute_softplus_f32(اعشاری درون_داد) {
    بازگشت (درون_داد > 20.0f) ? درون_داد : logf(1 + expf(درون_داد));
}
//
// logging
//

GGML_ATTRIBUTE_FORMAT(2, 3)
GGML_API تهی ggml_log_internal        (شمارش ggml_log_level level, ثابت نویسه * format, ...);
GGML_API تهی ggml_log_callback_default(شمارش ggml_log_level level, ثابت نویسه * text, تهی * user_data);

#define GGML_LOG(...)       ggml_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define GGML_LOG_INFO(...)  ggml_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define GGML_LOG_WARN(...)  ggml_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define GGML_LOG_ERROR(...) ggml_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GGML_LOG_DEBUG(...) ggml_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define GGML_LOG_CONT(...)  ggml_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

#define GGML_DEBUG 0

#اگر (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#وگرنه
#define GGML_PRINT_DEBUG(...)
#endif

#اگر (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) GGML_LOG_DEBUG(__VA_ARGS__)
#وگرنه
#define GGML_PRINT_DEBUG_5(...)
#endif

#اگر (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) GGML_LOG_DEBUG(__VA_ARGS__)
#وگرنه
#define GGML_PRINT_DEBUG_10(...)
#endif

// tensor params

ایستا تهی ggml_set_op_params(ساختار ggml_tensor * tensor, ثابت تهی * params, size_t params_size) {
    GGML_ASSERT(tensor != پوچ); // silence -Warray-bounds warnings
    أكد(params_size <= GGML_MAX_OP_PARAMS);
    memcpy(tensor->op_params, params, params_size);
}

ایستا int32_t ggml_get_op_params_i32(ثابت ساختار ggml_tensor * tensor, uint32_t i) {
    أكد(i < GGML_MAX_OP_PARAMS / اندازه(int32_t));
    بازگشت ((ثابت int32_t *)(tensor->op_params))[i];
}

ایستا اعشاری ggml_get_op_params_f32(ثابت ساختار ggml_tensor * tensor, uint32_t i) {
    أكد(i < GGML_MAX_OP_PARAMS / اندازه(اعشاری));
    بازگشت ((ثابت اعشاری *)(tensor->op_params))[i];
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
ایستا تهی ggml_set_op_params_i32(ساختار ggml_tensor * tensor, uint32_t i, int32_t مقدار) {
    أكد(i < GGML_MAX_OP_PARAMS / اندازه(int32_t));
    ((int32_t *)(tensor->op_params))[i] = مقدار;
}

ایستا تهی ggml_set_op_params_f32(ساختار ggml_tensor * tensor, uint32_t i, اعشاری مقدار) {
    أكد(i < GGML_MAX_OP_PARAMS / اندازه(اعشاری));
    ((اعشاری *)(tensor->op_params))[i] = مقدار;
}

ساختار ggml_map_custom1_op_params {
    ggml_custom1_op_t  fun;
    صحیح                n_tasks;
    تهی             * userdata;
};

ساختار ggml_map_custom2_op_params {
    ggml_custom2_op_t   fun;
    صحیح                 n_tasks;
    تهی              * userdata;
};

ساختار ggml_map_custom3_op_params {
    ggml_custom3_op_t fun;
    صحیح               n_tasks;
    تهی            * userdata;
};

ساختار ggml_custom_op_params {
    ggml_custom_op_t fun;
    صحیح              n_tasks;
    تهی           * userdata;
};

// bitset

تعریفنوع uint32_t ggml_bitset_t;

static_assert(اندازه(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#define BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#define BITSET_MASK (اندازه(ggml_bitset_t)*8 - 1)

ایستا size_t ggml_bitset_size(size_t n) {
    بازگشت (n + BITSET_MASK) >> BITSET_SHR;
}

ایستا درونخطی bool ggml_bitset_get(ثابت ggml_bitset_t * bitset, size_t i) {
    بازگشت !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
}

ایستا درونخطی تهی ggml_bitset_set(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_MASK));
}

ایستا درونخطی تهی ggml_bitset_clear(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_MASK));
}

// hash set

#define GGML_HASHSET_FULL ((size_t)-1)
#define GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)

ساختار ggml_hash_set {
    size_t size;
    ggml_bitset_t * used;       // whether or not the keys are in use i.e. set
    ساختار ggml_tensor ** کلیدها; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};

ساختار ggml_hash_set ggml_hash_set_new(size_t size);
تهی                 ggml_hash_set_free(ساختار ggml_hash_set * hash_set);

// returns the minimum size for a hash set that can hold min_sz elements
size_t ggml_hash_size(size_t min_sz);

// remove all elements from the hash set
تهی ggml_hash_set_reset(ساختار ggml_hash_set * hash_set);

// returns true if key is in the hash set
ایستا bool ggml_hash_contains(ثابت ساختار ggml_hash_set * hash_set, ساختار ggml_tensor * کلید);

// returns GGML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
ایستا size_t ggml_hash_find(ثابت ساختار ggml_hash_set * hash_set, ثابت ساختار ggml_tensor * کلید);

// returns GGML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
ایستا size_t ggml_hash_درج(ساختار ggml_hash_set * hash_set, ساختار ggml_tensor * کلید);

// return index, asserts if table is full
ایستا size_t ggml_hash_find_or_درج(ساختار ggml_hash_set * hash_set, ساختار ggml_tensor * کلید);

// hash function for ggml_tensor
ایستا درونخطی size_t ggml_hash(ثابت ساختار ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    بازگشت (size_t)(uintptr_t)p >> 4;
}

ایستا size_t ggml_hash_find(ثابت ساختار ggml_hash_set * hash_set, ثابت ساختار ggml_tensor * کلید) {
    size_t h = ggml_hash(کلید) % hash_set->size;

    // linear probing
    size_t i = h;
    وقتی (ggml_bitset_get(hash_set->used, i) && hash_set->کلیدها[i] != کلید) {
        i = (i + 1) % hash_set->size;
        اگر (i == h) {
            // visited all hash table entries -> not found
            بازگشت GGML_HASHSET_FULL;
        }
    }
    بازگشت i;
}

ایستا bool ggml_hash_contains(ثابت ساختار ggml_hash_set * hash_set, ساختار ggml_tensor * کلید) {
    size_t i = ggml_hash_find(hash_set, کلید);
    بازگشت i != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, i);
}

ایستا size_t ggml_hash_درج(ساختار ggml_hash_set * hash_set, ساختار ggml_tensor * کلید) {
    size_t h = ggml_hash(کلید) % hash_set->size;

    // linear probing
    size_t i = h;
    انجامبده {
        اگر (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->کلیدها[i] = کلید;
            بازگشت i;
        }
        اگر (hash_set->کلیدها[i] == کلید) {
            بازگشت GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_set->size;
    } وقتی (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

ایستا size_t ggml_hash_find_or_درج(ساختار ggml_hash_set * hash_set, ساختار ggml_tensor * کلید) {
    size_t h = ggml_hash(کلید) % hash_set->size;

    // linear probing
    size_t i = h;
    انجامبده {
        اگر (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->کلیدها[i] = کلید;
            بازگشت i;
        }
        اگر (hash_set->کلیدها[i] == کلید) {
            بازگشت i;
        }
        i = (i + 1) % hash_set->size;
    } وقتی (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

// computation graph

شمارش ggml_cgraph_eval_order {
    GGML_CGRAPH_EVAL_ORDER_چپ_TO_راست = 0,
    GGML_CGRAPH_EVAL_ORDER_راست_TO_چپ,
    GGML_CGRAPH_EVAL_ORDER_شمار
};

ساختار ggml_cgraph {
    صحیح size;    // maximum number of nodes/leafs/grads/grad_accs
    صحیح n_nodes; // number of nodes currently in use
    صحیح n_leafs; // number of leafs currently in use

    ساختار ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
    ساختار ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
    ساختار ggml_tensor ** grad_accs; // accumulators for node gradients
    ساختار ggml_tensor ** leafs;     // tensors with constant data
    int32_t             * use_counts;// number of uses of each tensor, indexed by hash table slot

    ساختار ggml_hash_set visited_hash_set;

    شمارش ggml_cgraph_eval_order order;

    // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
    // a value of 0 means it is not set and should be ignored
    uint64_t uid;
};

// returns a slice of cgraph with nodes [i0, i1)
// the slice does not have leafs or gradients
// if you need the gradients, get them from the original graph
ساختار ggml_cgraph ggml_graph_view(ساختار ggml_cgraph * cgraph, صحیح i0, صحیح i1);

// ggml-alloc.c: true if the operation can reuse memory from its sources
GGML_API bool ggml_op_can_inplace(شمارش ggml_op op);


// Memory allocation

GGML_API تهی * ggml_aligned_malloc(size_t size);
GGML_API تهی ggml_aligned_free(تهی * ptr, size_t size);

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

ایستا درونخطی اعشاری fp32_from_bits(uint32_t w) {
    اجتماع {
        uint32_t as_bits;
        اعشاری as_مقدار;
    } fp32;
    fp32.as_bits = w;
    بازگشت fp32.as_مقدار;
}

ایستا درونخطی uint32_t fp32_to_bits(اعشاری f) {
    اجتماع {
        اعشاری as_مقدار;
        uint32_t as_bits;
    } fp32;
    fp32.as_مقدار = f;
    بازگشت fp32.as_bits;
}

ایستا درونخطی اعشاری ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    ثابت uint32_t w = (uint32_t) h << 16;
    ثابت uint32_t sign = w & UINT32_C(0x80000000);
    ثابت uint32_t two_w = w + w;

    ثابت uint32_t exp_offset = UINT32_C(0xE0) << 23;
#اگر (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    ثابت اعشاری exp_scale = 0x1.0p-112f;
#وگرنه
    ثابت اعشاری exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    ثابت اعشاری normalized_مقدار = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    ثابت uint32_t magic_mask = UINT32_C(126) << 23;
    ثابت اعشاری magic_bias = 0.5f;
    ثابت اعشاری denormalized_مقدار = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    ثابت uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    ثابت uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_مقدار) : fp32_to_bits(normalized_مقدار));
    بازگشت fp32_from_bits(result);
}

ایستا درونخطی ggml_fp16_t ggml_compute_fp32_to_fp16(اعشاری f) {
#اگر (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    ثابت اعشاری scale_to_inf = 0x1.0p+112f;
    ثابت اعشاری scale_to_zero = 0x1.0p-110f;
#وگرنه
    ثابت اعشاری scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    ثابت اعشاری scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    اعشاری base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    ثابت uint32_t w = fp32_to_bits(f);
    ثابت uint32_t shl1_w = w + w;
    ثابت uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    اگر (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    ثابت uint32_t bits = fp32_to_bits(base);
    ثابت uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    ثابت uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    ثابت uint32_t nonsign = exp_bits + mantissa_bits;
    بازگشت (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

ایستا درونخطی اعشاری ggml_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;  // Stores the raw bit representation of the float

    // Handle special case for minimum exponent (denormalized float)
    اگر (x == 0) {
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
    وگرنه {
        // Construct normalized float by shifting exponent into position:
        // - Exponent field: 8 bits (positions 30-23)
        // - Mantissa: 0 (implicit leading 1)
        // Value = 2^(x - 127)
        bits = (uint32_t) x << 23;
    }

    اعشاری result;  // Final float value
                   // Safely reinterpret bit pattern as float without type-punning issues
    memcpy(&result, &bits, اندازه(اعشاری));
    بازگشت result;
}

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
ایستا درونخطی اعشاری ggml_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;

    // For x < 2: use precomputed denormal patterns
    اگر (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        bits = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    وگرنه {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        bits = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    اعشاری result;
    memcpy(&result, &bits, اندازه(اعشاری));
    بازگشت result;
}

#define GGML_E8M0_TO_FP32(x) ggml_e8m0_to_fp32(x)
#define GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
ایستا درونخطی اعشاری ggml_ue4m3_to_fp32(uint8_t x) {
    اگر (x == 0 || x == 0x7F) {
        بازگشت 0.0f;
    }
    صحیح   exp = (x >> 3) & 0xF;
    صحیح   man = x & 0x7;
    اعشاری raw;
    اگر (exp == 0) {
        raw = ldexpf((اعشاری) man, -9);
    } وگرنه {
        raw = ldexpf(1.0f + (اعشاری) man / 8.0f, exp - 7);
    }
    بازگشت raw * 0.5f;
}

ایستا درونخطی uint8_t ggml_fp32_to_ue4m3(اعشاری x) {
    اگر (!(x > 0.0f)) {
        بازگشت 0;
    }
    اگر (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t bits;
    memcpy(&bits, &x, 4);
    صحیح fp32_exp  = ((bits >> 23) & 0xFF) - 127;
    صحیح fp32_man  = (bits >> 20) & 0x7;
    صحیح ue4m3_exp = fp32_exp + 7;
    اگر (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        صحیح man = (صحیح) (x * 512.0f + 0.5f);
        اگر (man > 7) {
            man = 7;
        }
        اگر (man < 1) {
            بازگشت 0;
        }
        بازگشت (uint8_t) man;
    }
    اگر (ue4m3_exp >= 15) {
        بازگشت 0x7E;
    }
    صحیح round_bit = (bits >> 19) & 1;
    صحیح ue4m3_man = fp32_man + round_bit;
    اگر (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        اگر (ue4m3_exp >= 15) {
            بازگشت 0x7E;
        }
    }
    بازگشت (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
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
ایستا درونخطی اعشاری ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    اجتماع {
        اعشاری f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    بازگشت u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
ایستا درونخطی ggml_bf16_t ggml_compute_fp32_to_bf16(اعشاری s) {
    ggml_bf16_t h;
    اجتماع {
        اعشاری f;
        uint32_t i;
    } u;
    u.f = s;
    اگر ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.bits = (u.i >> 16) | 64; /* force to quiet */
        بازگشت h;
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    بازگشت h;
}

#define GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)
#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)

ایستا درونخطی int32_t ggml_گره_get_use_شمار(ثابت ساختار ggml_cgraph * cgraph, صحیح گره_idx) {
    ثابت ساختار ggml_tensor * گره = cgraph->nodes[گره_idx];

    size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, گره);
    اگر (!ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        بازگشت 0;
    }
    بازگشت cgraph->use_counts[hash_pos];
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
ایستا درونخطی bool ggml_گره_has_n_uses(ثابت ساختار ggml_cgraph * cgraph, صحیح گره_idx, int32_t n_uses) {
    ثابت ساختار ggml_tensor * گره = cgraph->nodes[گره_idx];

    // check the use count against how many we're replacing
    اگر (ggml_گره_get_use_شمار(cgraph, گره_idx) != n_uses) {
        بازگشت false;
    }

    // if node is a view, some other node might be using the intermediate result
    // via the view source.
    اگر (گره->view_src) {
        بازگشت false;
    }

    // If the user requested output for the node, can't fuse
    اگر (گره->flags & GGML_TENSOR_FLAG_OUTPUT) {
        بازگشت false;
    }

    بازگشت true;
}

// Returns true if nodes with indices { node_idxs } are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
ایستا درونخطی bool ggml_can_fuse_ext(ثابت ساختار ggml_cgraph * cgraph, ثابت صحیح * گره_idxs, ثابت شمارش ggml_op * ops, صحیح رقم_ops) {
    برای (صحیح i = 0; i < رقم_ops; ++i) {
        اگر (گره_idxs[i] >= cgraph->n_nodes) {
            بازگشت false;
        }

        ساختار ggml_tensor * گره = cgraph->nodes[گره_idxs[i]];
        اگر (گره->op != ops[i]) {
            بازگشت false;
        }
        اگر ((گره->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            بازگشت false;
        }
        اگر (i < رقم_ops - 1 && !ggml_گره_has_n_uses(cgraph, گره_idxs[i], 1)) {
            بازگشت false;
        }
        اگر (i > 0) {
            ساختار ggml_tensor * prev = cgraph->nodes[گره_idxs[i - 1]];
            اگر (گره->src[0] != prev && گره->src[1] != prev) {
                بازگشت false;
            }
            اگر (!ggml_are_same_shape(گره, prev)) {
                بازگشت false;
            }
        }
    }
    بازگشت true;
}

// same as above, for sequential indices starting at node_idx
ایستا درونخطی bool ggml_can_fuse(ثابت ساختار ggml_cgraph * cgraph, صحیح گره_idx, ثابت شمارش ggml_op * ops, صحیح رقم_ops) {
    أكد(رقم_ops < 32);

    اگر (گره_idx + رقم_ops > cgraph->n_nodes) {
        بازگشت false;
    }

    صحیح idxs[32];
    برای (صحیح i = 0; i < رقم_ops; ++i) {
        idxs[i] = گره_idx + i;
    }

    بازگشت ggml_can_fuse_ext(cgraph, idxs, ops, رقم_ops);
}

GGML_API bool ggml_can_fuse_subgraph_ext(ثابت ساختار ggml_cgraph * cgraph,
                                         ثابت صحیح *                گره_idxs,
                                         صحیح                        شمار,
                                         ثابت شمارش ggml_op *       ops,
                                         ثابت صحیح *                outputs,
                                         صحیح                        رقم_outputs);

// Returns true if the subgraph formed by {node_idxs} can be fused
// checks whethers all nodes which are not part of outputs can be elided
// by checking if their num_uses are confined to the subgraph
ایستا درونخطی bool ggml_can_fuse_subgraph(ثابت ساختار ggml_cgraph * cgraph,
                                          صحیح                        گره_idx,
                                          صحیح                        شمار,
                                          ثابت شمارش ggml_op *       ops,
                                          ثابت صحیح *                outputs,
                                          صحیح                        رقم_outputs) {
    GGML_ASSERT(شمار < 32);
    اگر (گره_idx + شمار > cgraph->n_nodes) {
        بازگشت false;
    }

    صحیح idxs[32];

    برای (صحیح i = 0; i < شمار; ++i) {
        idxs[i] = گره_idx + i;
    }

    بازگشت ggml_can_fuse_subgraph_ext(cgraph, idxs, شمار, ops, outputs, رقم_outputs);
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#شامل <array>
#شامل <initializer_list>
#شامل <vector>

// nicer C++ syntax for ggml_can_fuse
درونخطی bool ggml_can_fuse(ثابت ساختار ggml_cgraph * cgraph, صحیح گره_idx, std::initializer_list<شمارش ggml_op> ops) {
    بازگشت ggml_can_fuse(cgraph, گره_idx, ops.begin(), (صحیح)ops.size());
}

درونخطی bool ggml_can_fuse_subgraph(ثابت ساختار ggml_cgraph *          cgraph,
                                   صحیح                                 start_idx,
                                   std::initializer_list<شمارش ggml_op> ops,
                                   std::initializer_list<صحیح>          outputs = {}) {
    بازگشت ggml_can_fuse_subgraph(cgraph, start_idx, ops.size(), ops.begin(), outputs.begin(), outputs.size());
}

// Return true if the edges in the graph match expectations.
درونخطی bool ggml_check_edges(ثابت ساختار ggml_cgraph *                cgraph,
                             صحیح                                       start_idx,
                             std::initializer_list<std::array<صحیح, 3>> edges) {
    برای (ثابت auto & edge : edges) {
        صحیح dst_گره = edge[0];
        صحیح src_idx  = edge[1];
        صحیح src_گره = edge[2];
        اگر (cgraph->nodes[start_idx + dst_گره]->src[src_idx] != cgraph->nodes[start_idx + src_گره]) {
            بازگشت false;
        }
    }
    بازگشت true;
}

// expose GGUF internals for test code
GGML_API size_t gguf_type_size(شمارش gguf_type type);
GGML_API تهی gguf_write_to_buf(ثابت ساختار gguf_context * ctx, std::vector<int8_t> & buf, bool only_meta);
#endif // __cplusplus
