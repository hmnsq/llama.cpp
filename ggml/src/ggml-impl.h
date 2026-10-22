#ہدایت once

// GGML internal header

#شامل "ggml.h"
#شامل "gguf.h"

#شامل <assert.h>
#شامل <math.h>
#شامل <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#شامل <stdbool.h>
#شامل <stdint.h>
#شامل <string.h>

#اگر_تعریف_شدہ __ARM_FEATURE_SVE
#شامل <arm_sve.h>
#اگر_ختم // __ARM_FEATURE_SVE

#اگر defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#شامل <arm_neon.h>
#اگر_ختم

#اگر_تعریف_شدہ __cplusplus
خارجی "C" {
#اگر_ختم

خالی ggml_print_backtrace(خالی);

uint64_t ggml_گراف_اگلا_uid(خالی);

#اگر_غیر_تعریف_شدہ MIN
#    define MIN(a, b) ((a) < (b) ? (a) : (b))
#اگر_ختم

#اگر_غیر_تعریف_شدہ MAX
#    define MAX(a, b) ((a) > (b) ? (a) : (b))
#اگر_ختم

// required for mmap as gguf only guarantees 32-byte alignment
#تعریف_کرو TENSOR_ALIGNMENT 32

// static_assert should be a #define, but if it's not,
// fall back to the _Static_assert C11 keyword.
// if C99 - static_assert is noop
// ref: https://stackoverflow.com/a/53923785/4039976
#اگر_غیر_تعریف_شدہ __cplusplus
    #اگر_غیر_تعریف_شدہ static_assert
        #اگر defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
            #تعریف_کرو static_assert(cond, پیغام_2) _Static_assert(cond, پیغام_2)
        #ورنہ
            #تعریف_کرو static_assert(cond, پیغام_2) ساخت global_scope_noop_trick
        #اگر_ختم
    #اگر_ختم
#اگر_ختم

ساکن اندرونی صحیح ggml_up32(صحیح n) {
    واپس (n + 31) & ~31;
}

//static inline int ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

ساکن اندرونی صحیح ggml_اوپر(صحیح n, صحیح m) {
    // assert m is a power of 2
    GGML_ASSERT((m & (m - 1)) == 0);
    واپس (n + m - 1) & ~(m - 1);
}

// TODO: move to ggml.h? (won't be able to inline)
ساکن bool ggml_are_same_layout(ثابت ساخت ggml_tensor * a, ثابت ساخت ggml_tensor * b) {
    اگر (a->type != b->type) {
        واپس false;
    }
    برائے (صحیح i = 0; i < GGML_MAX_DIMS; i++) {
        اگر (a->ne[i] != b->ne[i]) {
            واپس false;
        }
        اگر (a->nb[i] != b->nb[i]) {
            واپس false;
        }
    }
    واپس true;
}

ساکن bool ggml_op_is_خالی_خانے(شمار ggml_op op) {
    انتخاب (op) {
        حالت GGML_OP_NONE:
        حالت GGML_OP_RESHAPE:
        حالت GGML_OP_TRANSPOSE:
        حالت GGML_OP_VIEW:
        حالت GGML_OP_ترتیب_بدلو:
            واپس true;
        طےشدہ:
            واپس false;
    }
}

ساکن اندرونی bool ggml_impl_is_view(ثابت ساخت ggml_tensor * t) {
    واپس t->view_src != تہی;
}

ساکن اندرونی اعشاری ggml_compute_softplus_f32(اعشاری داخلہ) {
    واپس (داخلہ > 20.0f) ? داخلہ : logf(1 + expf(داخلہ));
}
//
// logging
//

GGML_ATTRIBUTE_FORMAT(2, 3)
GGML_API خالی ggml_log_اندرونی        (شمار ggml_log_level level, ثابت حرف * format, ...);
GGML_API خالی ggml_log_callback_default(شمار ggml_log_level level, ثابت حرف * متن_2, خالی * user_ڈیٹا);

#تعریف_کرو GGML_LOG(...)       ggml_log_اندرونی(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#تعریف_کرو GGML_LOG_INFO(...)  ggml_log_اندرونی(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#تعریف_کرو GGML_LOG_WARN(...)  ggml_log_اندرونی(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#تعریف_کرو GGML_LOG_ERROR(...) ggml_log_اندرونی(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#تعریف_کرو GGML_LOG_DEBUG(...) ggml_log_اندرونی(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#تعریف_کرو GGML_LOG_CONT(...)  ggml_log_اندرونی(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

#تعریف_کرو GGML_DEBUG 0

#اگر (GGML_DEBUG >= 1)
#تعریف_کرو GGML_PRINT_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#ورنہ
#تعریف_کرو GGML_PRINT_DEBUG(...)
#اگر_ختم

#اگر (GGML_DEBUG >= 5)
#تعریف_کرو GGML_PRINT_DEBUG_5(...) GGML_LOG_DEBUG(__VA_ARGS__)
#ورنہ
#تعریف_کرو GGML_PRINT_DEBUG_5(...)
#اگر_ختم

#اگر (GGML_DEBUG >= 10)
#تعریف_کرو GGML_PRINT_DEBUG_10(...) GGML_LOG_DEBUG(__VA_ARGS__)
#ورنہ
#تعریف_کرو GGML_PRINT_DEBUG_10(...)
#اگر_ختم

// tensor params

ساکن خالی ggml_سیٹ_op_params(ساخت ggml_tensor * tensor, ثابت خالی * params, size_t params_سائز) {
    GGML_ASSERT(tensor != تہی); // silence -Warray-bounds warnings
    أكد(params_سائز <= GGML_MAX_OP_PARAMS);
    حافظہ_نقل(tensor->op_params, params, params_سائز);
}

ساکن int32_t ggml_get_op_params_i32(ثابت ساخت ggml_tensor * tensor, uint32_t i) {
    أكد(i < GGML_MAX_OP_PARAMS / حجم(int32_t));
    واپس ((ثابت int32_t *)(tensor->op_params))[i];
}

ساکن اعشاری ggml_get_op_params_f32(ثابت ساخت ggml_tensor * tensor, uint32_t i) {
    أكد(i < GGML_MAX_OP_PARAMS / حجم(اعشاری));
    واپس ((ثابت اعشاری *)(tensor->op_params))[i];
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
ساکن خالی ggml_سیٹ_op_params_i32(ساخت ggml_tensor * tensor, uint32_t i, int32_t قدر) {
    أكد(i < GGML_MAX_OP_PARAMS / حجم(int32_t));
    ((int32_t *)(tensor->op_params))[i] = قدر;
}

ساکن خالی ggml_سیٹ_op_params_f32(ساخت ggml_tensor * tensor, uint32_t i, اعشاری قدر) {
    أكد(i < GGML_MAX_OP_PARAMS / حجم(اعشاری));
    ((اعشاری *)(tensor->op_params))[i] = قدر;
}

ساخت ggml_map_custom1_op_params {
    ggml_custom1_op_t  fun;
    صحیح                n_tasks;
    خالی             * userdata;
};

ساخت ggml_map_custom2_op_params {
    ggml_custom2_op_t   fun;
    صحیح                 n_tasks;
    خالی              * userdata;
};

ساخت ggml_map_custom3_op_params {
    ggml_custom3_op_t fun;
    صحیح               n_tasks;
    خالی            * userdata;
};

ساخت ggml_custom_op_params {
    ggml_custom_op_t fun;
    صحیح              n_tasks;
    خالی           * userdata;
};

// bitset

قسمتعریف uint32_t ggml_bitset_t;

static_assert(حجم(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#تعریف_کرو BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#تعریف_کرو BITSET_نقاب (حجم(ggml_bitset_t)*8 - 1)

ساکن size_t ggml_bitset_سائز(size_t n) {
    واپس (n + BITSET_نقاب) >> BITSET_SHR;
}

ساکن اندرونی bool ggml_bitset_get(ثابت ggml_bitset_t * bitset, size_t i) {
    واپس !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_نقاب)));
}

ساکن اندرونی خالی ggml_bitset_سیٹ(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_نقاب));
}

ساکن اندرونی خالی ggml_bitset_clear(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_نقاب));
}

// hash set

#تعریف_کرو GGML_HASHSET_بھرا ((size_t)-1)
#تعریف_کرو GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)

ساخت ggml_hash_سیٹ {
    size_t سائز;
    ggml_bitset_t * مستعمل;       // whether or not the keys are in use i.e. set
    ساخت ggml_tensor ** کلیدیں; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};

ساخت ggml_hash_سیٹ ggml_hash_سیٹ_new(size_t سائز);
خالی                 ggml_hash_سیٹ_free(ساخت ggml_hash_سیٹ * hash_سیٹ);

// returns the minimum size for a hash set that can hold min_sz elements
size_t ggml_hash_سائز(size_t min_sz);

// remove all elements from the hash set
خالی ggml_hash_سیٹ_reset(ساخت ggml_hash_سیٹ * hash_سیٹ);

// returns true if key is in the hash set
ساکن bool ggml_hash_شامل_ہے(ثابت ساخت ggml_hash_سیٹ * hash_سیٹ, ساخت ggml_tensor * کلید);

// returns GGML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
ساکن size_t ggml_hash_ڈھونڈو(ثابت ساخت ggml_hash_سیٹ * hash_سیٹ, ثابت ساخت ggml_tensor * کلید);

// returns GGML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
ساکن size_t ggml_hash_داخل_کرو(ساخت ggml_hash_سیٹ * hash_سیٹ, ساخت ggml_tensor * کلید);

// return index, asserts if table is full
ساکن size_t ggml_hash_ڈھونڈو_or_داخل_کرو(ساخت ggml_hash_سیٹ * hash_سیٹ, ساخت ggml_tensor * کلید);

// hash function for ggml_tensor
ساکن اندرونی size_t ggml_hash(ثابت ساخت ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    واپس (size_t)(uintptr_t)p >> 4;
}

ساکن size_t ggml_hash_ڈھونڈو(ثابت ساخت ggml_hash_سیٹ * hash_سیٹ, ثابت ساخت ggml_tensor * کلید) {
    size_t h = ggml_hash(کلید) % hash_سیٹ->سائز;

    // linear probing
    size_t i = h;
    جب (ggml_bitset_get(hash_سیٹ->مستعمل, i) && hash_سیٹ->کلیدیں[i] != کلید) {
        i = (i + 1) % hash_سیٹ->سائز;
        اگر (i == h) {
            // visited all hash table entries -> not found
            واپس GGML_HASHSET_بھرا;
        }
    }
    واپس i;
}

ساکن bool ggml_hash_شامل_ہے(ثابت ساخت ggml_hash_سیٹ * hash_سیٹ, ساخت ggml_tensor * کلید) {
    size_t i = ggml_hash_ڈھونڈو(hash_سیٹ, کلید);
    واپس i != GGML_HASHSET_بھرا && ggml_bitset_get(hash_سیٹ->مستعمل, i);
}

ساکن size_t ggml_hash_داخل_کرو(ساخت ggml_hash_سیٹ * hash_سیٹ, ساخت ggml_tensor * کلید) {
    size_t h = ggml_hash(کلید) % hash_سیٹ->سائز;

    // linear probing
    size_t i = h;
    کرو {
        اگر (!ggml_bitset_get(hash_سیٹ->مستعمل, i)) {
            ggml_bitset_سیٹ(hash_سیٹ->مستعمل, i);
            hash_سیٹ->کلیدیں[i] = کلید;
            واپس i;
        }
        اگر (hash_سیٹ->کلیدیں[i] == کلید) {
            واپس GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_سیٹ->سائز;
    } جب (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

ساکن size_t ggml_hash_ڈھونڈو_or_داخل_کرو(ساخت ggml_hash_سیٹ * hash_سیٹ, ساخت ggml_tensor * کلید) {
    size_t h = ggml_hash(کلید) % hash_سیٹ->سائز;

    // linear probing
    size_t i = h;
    کرو {
        اگر (!ggml_bitset_get(hash_سیٹ->مستعمل, i)) {
            ggml_bitset_سیٹ(hash_سیٹ->مستعمل, i);
            hash_سیٹ->کلیدیں[i] = کلید;
            واپس i;
        }
        اگر (hash_سیٹ->کلیدیں[i] == کلید) {
            واپس i;
        }
        i = (i + 1) % hash_سیٹ->سائز;
    } جب (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

// computation graph

شمار ggml_cgraph_eval_ترتیب {
    GGML_CGRAPH_EVAL_ترتیب_بایاں_مقصد_دایاں = 0,
    GGML_CGRAPH_EVAL_ترتیب_دایاں_مقصد_بایاں,
    GGML_CGRAPH_EVAL_ترتیب_گنتی
};

ساخت ggml_cgraph {
    صحیح سائز;    // maximum number of nodes/leafs/grads/grad_accs
    صحیح n_nodes; // number of nodes currently in use
    صحیح n_leafs; // number of leafs currently in use

    ساخت ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
    ساخت ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
    ساخت ggml_tensor ** grad_accs; // accumulators for node gradients
    ساخت ggml_tensor ** leafs;     // tensors with constant data
    int32_t             * use_counts;// number of uses of each tensor, indexed by hash table slot

    ساخت ggml_hash_سیٹ دیکھا_hash_سیٹ;

    شمار ggml_cgraph_eval_ترتیب ترتیب;

    // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
    // a value of 0 means it is not set and should be ignored
    uint64_t uid;
};

// returns a slice of cgraph with nodes [i0, i1)
// the slice does not have leafs or gradients
// if you need the gradients, get them from the original graph
ساخت ggml_cgraph ggml_گراف_view(ساخت ggml_cgraph * cgraph, صحیح i0, صحیح i1);

// ggml-alloc.c: true if the operation can reuse memory from its sources
GGML_API bool ggml_op_can_inplace(شمار ggml_op op);


// Memory allocation

GGML_API خالی * ggml_aligned_malloc(size_t سائز);
GGML_API خالی ggml_aligned_free(خالی * ptr, size_t سائز);

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

ساکن اندرونی اعشاری fp32_from_بٹ(uint32_t w) {
    اتحاد {
        uint32_t as_بٹ;
        اعشاری as_قدر;
    } fp32;
    fp32.as_بٹ = w;
    واپس fp32.as_قدر;
}

ساکن اندرونی uint32_t fp32_مقصد_بٹ(اعشاری f) {
    اتحاد {
        اعشاری as_قدر;
        uint32_t as_بٹ;
    } fp32;
    fp32.as_قدر = f;
    واپس fp32.as_بٹ;
}

ساکن اندرونی اعشاری ggml_compute_fp16_مقصد_fp32(ggml_fp16_t h) {
    ثابت uint32_t w = (uint32_t) h << 16;
    ثابت uint32_t sign = w & UINT32_C(0x80000000);
    ثابت uint32_t two_w = w + w;

    ثابت uint32_t exp_انحراف = UINT32_C(0xE0) << 23;
#اگر (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    ثابت اعشاری exp_scale = 0x1.0p-112f;
#ورنہ
    ثابت اعشاری exp_scale = fp32_from_بٹ(UINT32_C(0x7800000));
#اگر_ختم
    ثابت اعشاری normalized_قدر = fp32_from_بٹ((two_w >> 4) + exp_انحراف) * exp_scale;

    ثابت uint32_t magic_نقاب = UINT32_C(126) << 23;
    ثابت اعشاری magic_bias = 0.5f;
    ثابت اعشاری denormalized_قدر = fp32_from_بٹ((two_w >> 17) | magic_نقاب) - magic_bias;

    ثابت uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    ثابت uint32_t نتیجہ = sign |
        (two_w < denormalized_cutoff ? fp32_مقصد_بٹ(denormalized_قدر) : fp32_مقصد_بٹ(normalized_قدر));
    واپس fp32_from_بٹ(نتیجہ);
}

ساکن اندرونی ggml_fp16_t ggml_compute_fp32_مقصد_fp16(اعشاری f) {
#اگر (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    ثابت اعشاری scale_مقصد_inf = 0x1.0p+112f;
    ثابت اعشاری scale_مقصد_zero = 0x1.0p-110f;
#ورنہ
    ثابت اعشاری scale_مقصد_inf = fp32_from_بٹ(UINT32_C(0x77800000));
    ثابت اعشاری scale_مقصد_zero = fp32_from_بٹ(UINT32_C(0x08800000));
#اگر_ختم
    اعشاری بنیاد = (fabsf(f) * scale_مقصد_inf) * scale_مقصد_zero;

    ثابت uint32_t w = fp32_مقصد_بٹ(f);
    ثابت uint32_t shl1_w = w + w;
    ثابت uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    اگر (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    بنیاد = fp32_from_بٹ((bias >> 1) + UINT32_C(0x07800000)) + بنیاد;
    ثابت uint32_t بٹ = fp32_مقصد_بٹ(بنیاد);
    ثابت uint32_t exp_بٹ = (بٹ >> 13) & UINT32_C(0x00007C00);
    ثابت uint32_t mantissa_بٹ = بٹ & UINT32_C(0x00000FFF);
    ثابت uint32_t nonsign = exp_بٹ + mantissa_بٹ;
    واپس (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#تعریف_کرو GGML_COMPUTE_FP16_مقصد_FP32(x) ggml_compute_fp16_مقصد_fp32(x)
#تعریف_کرو GGML_COMPUTE_FP32_مقصد_FP16(x) ggml_compute_fp32_مقصد_fp16(x)

#تعریف_کرو GGML_FP16_مقصد_FP32(x) GGML_COMPUTE_FP16_مقصد_FP32(x)
#تعریف_کرو GGML_FP32_مقصد_FP16(x) GGML_COMPUTE_FP32_مقصد_FP16(x)

ساکن اندرونی اعشاری ggml_e8m0_مقصد_fp32(uint8_t x) {
    uint32_t بٹ;  // Stores the raw bit representation of the float

    // Handle special case for minimum exponent (denormalized float)
    اگر (x == 0) {
        // Bit pattern for 2^(-127):
        // - Sign bit: 0 (positive)
        // - Exponent: 0 (denormalized number)
        // - Mantissa: 0x400000 (0.5 in fractional form)
        // Value = 0.5 * 2^(-126) = 2^(-127)
        بٹ = 0x00400000;
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
    ورنہ {
        // Construct normalized float by shifting exponent into position:
        // - Exponent field: 8 bits (positions 30-23)
        // - Mantissa: 0 (implicit leading 1)
        // Value = 2^(x - 127)
        بٹ = (uint32_t) x << 23;
    }

    اعشاری نتیجہ;  // Final float value
                   // Safely reinterpret bit pattern as float without type-punning issues
    حافظہ_نقل(&نتیجہ, &بٹ, حجم(اعشاری));
    واپس نتیجہ;
}

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
ساکن اندرونی اعشاری ggml_e8m0_مقصد_fp32_half(uint8_t x) {
    uint32_t بٹ;

    // For x < 2: use precomputed denormal patterns
    اگر (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        بٹ = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    ورنہ {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        بٹ = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    اعشاری نتیجہ;
    حافظہ_نقل(&نتیجہ, &بٹ, حجم(اعشاری));
    واپس نتیجہ;
}

#تعریف_کرو GGML_E8_M0_مقصد_FP32(x) ggml_e8m0_مقصد_fp32(x)
#تعریف_کرو GGML_E8_M0_مقصد_FP32_HALF(x) ggml_e8m0_مقصد_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
ساکن اندرونی اعشاری ggml_ue4m3_مقصد_fp32(uint8_t x) {
    اگر (x == 0 || x == 0x7F) {
        واپس 0.0f;
    }
    صحیح   قوت_نما = (x >> 3) & 0xF;
    صحیح   man = x & 0x7;
    اعشاری raw;
    اگر (قوت_نما == 0) {
        raw = ldexpf((اعشاری) man, -9);
    } ورنہ {
        raw = ldexpf(1.0f + (اعشاری) man / 8.0f, قوت_نما - 7);
    }
    واپس raw * 0.5f;
}

ساکن اندرونی uint8_t ggml_fp32_مقصد_ue4m3(اعشاری x) {
    اگر (!(x > 0.0f)) {
        واپس 0;
    }
    اگر (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t بٹ;
    حافظہ_نقل(&بٹ, &x, 4);
    صحیح fp32_exp  = ((بٹ >> 23) & 0xFF) - 127;
    صحیح fp32_man  = (بٹ >> 20) & 0x7;
    صحیح ue4m3_exp = fp32_exp + 7;
    اگر (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        صحیح man = (صحیح) (x * 512.0f + 0.5f);
        اگر (man > 7) {
            man = 7;
        }
        اگر (man < 1) {
            واپس 0;
        }
        واپس (uint8_t) man;
    }
    اگر (ue4m3_exp >= 15) {
        واپس 0x7E;
    }
    صحیح باری_ثنائی_ہندسہ = (بٹ >> 19) & 1;
    صحیح ue4m3_man = fp32_man + باری_ثنائی_ہندسہ;
    اگر (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        اگر (ue4m3_exp >= 15) {
            واپس 0x7E;
        }
    }
    واپس (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
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
ساکن اندرونی اعشاری ggml_compute_bf16_مقصد_fp32(ggml_bf16_t h) {
    اتحاد {
        اعشاری f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.بٹ << 16;
    واپس u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
ساکن اندرونی ggml_bf16_t ggml_compute_fp32_مقصد_bf16(اعشاری s) {
    ggml_bf16_t h;
    اتحاد {
        اعشاری f;
        uint32_t i;
    } u;
    u.f = s;
    اگر ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.بٹ = (u.i >> 16) | 64; /* force to quiet */
        واپس h;
    }
    h.بٹ = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    واپس h;
}

#تعریف_کرو GGML_FP32_مقصد_BF16(x) ggml_compute_fp32_مقصد_bf16(x)
#تعریف_کرو GGML_BF16_مقصد_FP32(x) ggml_compute_bf16_مقصد_fp32(x)

ساکن اندرونی int32_t ggml_گرہ_get_use_گنتی(ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx) {
    ثابت ساخت ggml_tensor * گرہ = cgraph->nodes[گرہ_idx];

    size_t hash_pos = ggml_hash_ڈھونڈو(&cgraph->دیکھا_hash_سیٹ, گرہ);
    اگر (!ggml_bitset_get(cgraph->دیکھا_hash_سیٹ.مستعمل, hash_pos)) {
        واپس 0;
    }
    واپس cgraph->use_counts[hash_pos];
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
ساکن اندرونی bool ggml_گرہ_has_n_استعمال_کرتا(ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx, int32_t n_استعمال_کرتا) {
    ثابت ساخت ggml_tensor * گرہ = cgraph->nodes[گرہ_idx];

    // check the use count against how many we're replacing
    اگر (ggml_گرہ_get_use_گنتی(cgraph, گرہ_idx) != n_استعمال_کرتا) {
        واپس false;
    }

    // if node is a view, some other node might be using the intermediate result
    // via the view source.
    اگر (گرہ->view_src) {
        واپس false;
    }

    // If the user requested output for the node, can't fuse
    اگر (گرہ->flags & GGML_TENSOR_FLAG_OUTPUT) {
        واپس false;
    }

    واپس true;
}

// Returns true if nodes with indices { node_idxs } are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
ساکن اندرونی bool ggml_can_fuse_ext(ثابت ساخت ggml_cgraph * cgraph, ثابت صحیح * گرہ_idxs, ثابت شمار ggml_op * ops, صحیح عدد_ops) {
    برائے (صحیح i = 0; i < عدد_ops; ++i) {
        اگر (گرہ_idxs[i] >= cgraph->n_nodes) {
            واپس false;
        }

        ساخت ggml_tensor * گرہ = cgraph->nodes[گرہ_idxs[i]];
        اگر (گرہ->op != ops[i]) {
            واپس false;
        }
        اگر ((گرہ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            واپس false;
        }
        اگر (i < عدد_ops - 1 && !ggml_گرہ_has_n_استعمال_کرتا(cgraph, گرہ_idxs[i], 1)) {
            واپس false;
        }
        اگر (i > 0) {
            ساخت ggml_tensor * پچھلا = cgraph->nodes[گرہ_idxs[i - 1]];
            اگر (گرہ->src[0] != پچھلا && گرہ->src[1] != پچھلا) {
                واپس false;
            }
            اگر (!ggml_are_same_shape(گرہ, پچھلا)) {
                واپس false;
            }
        }
    }
    واپس true;
}

// same as above, for sequential indices starting at node_idx
ساکن اندرونی bool ggml_can_fuse(ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx, ثابت شمار ggml_op * ops, صحیح عدد_ops) {
    أكد(عدد_ops < 32);

    اگر (گرہ_idx + عدد_ops > cgraph->n_nodes) {
        واپس false;
    }

    صحیح idxs[32];
    برائے (صحیح i = 0; i < عدد_ops; ++i) {
        idxs[i] = گرہ_idx + i;
    }

    واپس ggml_can_fuse_ext(cgraph, idxs, ops, عدد_ops);
}

GGML_API bool ggml_can_fuse_subgraph_ext(ثابت ساخت ggml_cgraph * cgraph,
                                         ثابت صحیح *                گرہ_idxs,
                                         صحیح                        گنتی,
                                         ثابت شمار ggml_op *       ops,
                                         ثابت صحیح *                outputs,
                                         صحیح                        عدد_outputs);

// Returns true if the subgraph formed by {node_idxs} can be fused
// checks whethers all nodes which are not part of outputs can be elided
// by checking if their num_uses are confined to the subgraph
ساکن اندرونی bool ggml_can_fuse_subgraph(ثابت ساخت ggml_cgraph * cgraph,
                                          صحیح                        گرہ_idx,
                                          صحیح                        گنتی,
                                          ثابت شمار ggml_op *       ops,
                                          ثابت صحیح *                outputs,
                                          صحیح                        عدد_outputs) {
    GGML_ASSERT(گنتی < 32);
    اگر (گرہ_idx + گنتی > cgraph->n_nodes) {
        واپس false;
    }

    صحیح idxs[32];

    برائے (صحیح i = 0; i < گنتی; ++i) {
        idxs[i] = گرہ_idx + i;
    }

    واپس ggml_can_fuse_subgraph_ext(cgraph, idxs, گنتی, ops, outputs, عدد_outputs);
}

#اگر_تعریف_شدہ __cplusplus
}
#اگر_ختم

#اگر_تعریف_شدہ __cplusplus
#شامل <array>
#شامل <initializer_list>
#شامل <vector>

// nicer C++ syntax for ggml_can_fuse
اندرونی bool ggml_can_fuse(ثابت ساخت ggml_cgraph * cgraph, صحیح گرہ_idx, std::initializer_list<شمار ggml_op> ops) {
    واپس ggml_can_fuse(cgraph, گرہ_idx, ops.begin(), (صحیح)ops.سائز());
}

اندرونی bool ggml_can_fuse_subgraph(ثابت ساخت ggml_cgraph *          cgraph,
                                   صحیح                                 آغاز_idx,
                                   std::initializer_list<شمار ggml_op> ops,
                                   std::initializer_list<صحیح>          outputs = {}) {
    واپس ggml_can_fuse_subgraph(cgraph, آغاز_idx, ops.سائز(), ops.begin(), outputs.begin(), outputs.سائز());
}

// Return true if the edges in the graph match expectations.
اندرونی bool ggml_check_کنارے(ثابت ساخت ggml_cgraph *                cgraph,
                             صحیح                                       آغاز_idx,
                             std::initializer_list<std::array<صحیح, 3>> کنارے) {
    برائے (ثابت خودکار & کنارہ : کنارے) {
        صحیح dst_گرہ = کنارہ[0];
        صحیح src_idx  = کنارہ[1];
        صحیح src_گرہ = کنارہ[2];
        اگر (cgraph->nodes[آغاز_idx + dst_گرہ]->src[src_idx] != cgraph->nodes[آغاز_idx + src_گرہ]) {
            واپس false;
        }
    }
    واپس true;
}

// expose GGUF internals for test code
GGML_API size_t gguf_type_سائز(شمار gguf_type type);
GGML_API خالی gguf_write_مقصد_buf(ثابت ساخت gguf_context * ctx, std::سمتیہ<int8_t> & buf, bool only_meta);
#اگر_ختم // __cplusplus
