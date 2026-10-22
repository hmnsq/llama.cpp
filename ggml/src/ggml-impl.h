#لارښوونه once

// GGML internal header

#ضمن "ggml.h"
#ضمن "gguf.h"

#ضمن <assert.h>
#ضمن <math.h>
#ضمن <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#ضمن <stdbool.h>
#ضمن <stdint.h>
#ضمن <string.h>

#که_تعریف_شوی __ARM_FEATURE_SVE
#ضمن <arm_sve.h>
#د_که_پای // __ARM_FEATURE_SVE

#إذا defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#ضمن <arm_neon.h>
#د_که_پای

#که_تعریف_شوی __cplusplus
خارجي "C" {
#د_که_پای

فراغ ggml_print_backtrace(فراغ);

uint64_t ggml_ګراف_بل_uid(فراغ);

#که_نه_تعریف_شوی MIN
#    define MIN(a, b) ((a) < (b) ? (a) : (b))
#د_که_پای

#که_نه_تعریف_شوی MAX
#    define MAX(a, b) ((a) > (b) ? (a) : (b))
#د_که_پای

// required for mmap as gguf only guarantees 32-byte alignment
#تعریف_کړه TENSOR_ALIGNMENT 32

// static_assert should be a #define, but if it's not,
// fall back to the _Static_assert C11 keyword.
// if C99 - static_assert is noop
// ref: https://stackoverflow.com/a/53923785/4039976
#که_نه_تعریف_شوی __cplusplus
    #که_نه_تعریف_شوی static_assert
        #إذا defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
            #تعریف_کړه static_assert(cond, پیغام_2) _Static_assert(cond, پیغام_2)
        #وإلا
            #تعریف_کړه static_assert(cond, پیغام_2) بنية global_scope_noop_trick
        #د_که_پای
    #د_که_پای
#د_که_پای

ساكن مضمن صحيح ggml_up32(صحيح n) {
    أرجع (n + 31) & ~31;
}

//static inline int ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

ساكن مضمن صحيح ggml_پورته(صحيح n, صحيح m) {
    // assert m is a power of 2
    GGML_ASSERT((m & (m - 1)) == 0);
    أرجع (n + m - 1) & ~(m - 1);
}

// TODO: move to ggml.h? (won't be able to inline)
ساكن bool ggml_are_same_layout(ثابت بنية ggml_tensor * a, ثابت بنية ggml_tensor * b) {
    إذا (a->type != b->type) {
        أرجع false;
    }
    لكل (صحيح i = 0; i < GGML_MAX_DIMS; i++) {
        إذا (a->ne[i] != b->ne[i]) {
            أرجع false;
        }
        إذا (a->nb[i] != b->nb[i]) {
            أرجع false;
        }
    }
    أرجع true;
}

ساكن bool ggml_op_is_تشه(تعداد ggml_op op) {
    بدل (op) {
        حالة GGML_OP_NONE:
        حالة GGML_OP_RESHAPE:
        حالة GGML_OP_TRANSPOSE:
        حالة GGML_OP_VIEW:
        حالة GGML_OP_ترتیب_بدلول:
            أرجع true;
        افتراضي:
            أرجع false;
    }
}

ساكن مضمن bool ggml_impl_is_view(ثابت بنية ggml_tensor * t) {
    أرجع t->view_src != عدم;
}

ساكن مضمن عائم ggml_compute_softplus_f32(عائم ننوت) {
    أرجع (ننوت > 20.0f) ? ننوت : logf(1 + expf(ننوت));
}
//
// logging
//

GGML_ATTRIBUTE_FORMAT(2, 3)
GGML_API فراغ ggml_log_دننی        (تعداد ggml_log_level level, ثابت حرف * format, ...);
GGML_API فراغ ggml_log_callback_default(تعداد ggml_log_level level, ثابت حرف * متن_2, فراغ * user_معلومات);

#تعریف_کړه GGML_LOG(...)       ggml_log_دننی(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#تعریف_کړه GGML_LOG_INFO(...)  ggml_log_دننی(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#تعریف_کړه GGML_LOG_WARN(...)  ggml_log_دننی(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#تعریف_کړه GGML_LOG_ERROR(...) ggml_log_دننی(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#تعریف_کړه GGML_LOG_DEBUG(...) ggml_log_دننی(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#تعریف_کړه GGML_LOG_CONT(...)  ggml_log_دننی(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

#تعریف_کړه GGML_DEBUG 0

#إذا (GGML_DEBUG >= 1)
#تعریف_کړه GGML_PRINT_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#وإلا
#تعریف_کړه GGML_PRINT_DEBUG(...)
#د_که_پای

#إذا (GGML_DEBUG >= 5)
#تعریف_کړه GGML_PRINT_DEBUG_5(...) GGML_LOG_DEBUG(__VA_ARGS__)
#وإلا
#تعریف_کړه GGML_PRINT_DEBUG_5(...)
#د_که_پای

#إذا (GGML_DEBUG >= 10)
#تعریف_کړه GGML_PRINT_DEBUG_10(...) GGML_LOG_DEBUG(__VA_ARGS__)
#وإلا
#تعریف_کړه GGML_PRINT_DEBUG_10(...)
#د_که_پای

// tensor params

ساكن فراغ ggml_سېټ_op_params(بنية ggml_tensor * tensor, ثابت فراغ * params, size_t params_اندازه) {
    GGML_ASSERT(tensor != عدم); // silence -Warray-bounds warnings
    أكد(params_اندازه <= GGML_MAX_OP_PARAMS);
    حافظه_نقل(tensor->op_params, params, params_اندازه);
}

ساكن int32_t ggml_get_op_params_i32(ثابت بنية ggml_tensor * tensor, uint32_t i) {
    أكد(i < GGML_MAX_OP_PARAMS / حجم(int32_t));
    أرجع ((ثابت int32_t *)(tensor->op_params))[i];
}

ساكن عائم ggml_get_op_params_f32(ثابت بنية ggml_tensor * tensor, uint32_t i) {
    أكد(i < GGML_MAX_OP_PARAMS / حجم(عائم));
    أرجع ((ثابت عائم *)(tensor->op_params))[i];
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
ساكن فراغ ggml_سېټ_op_params_i32(بنية ggml_tensor * tensor, uint32_t i, int32_t ارزښت) {
    أكد(i < GGML_MAX_OP_PARAMS / حجم(int32_t));
    ((int32_t *)(tensor->op_params))[i] = ارزښت;
}

ساكن فراغ ggml_سېټ_op_params_f32(بنية ggml_tensor * tensor, uint32_t i, عائم ارزښت) {
    أكد(i < GGML_MAX_OP_PARAMS / حجم(عائم));
    ((عائم *)(tensor->op_params))[i] = ارزښت;
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
#تعریف_کړه BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#تعریف_کړه BITSET_ماسک (حجم(ggml_bitset_t)*8 - 1)

ساكن size_t ggml_bitset_اندازه(size_t n) {
    أرجع (n + BITSET_ماسک) >> BITSET_SHR;
}

ساكن مضمن bool ggml_bitset_get(ثابت ggml_bitset_t * bitset, size_t i) {
    أرجع !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_ماسک)));
}

ساكن مضمن فراغ ggml_bitset_سېټ(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_ماسک));
}

ساكن مضمن فراغ ggml_bitset_clear(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_ماسک));
}

// hash set

#تعریف_کړه GGML_HASHSET_ډک ((size_t)-1)
#تعریف_کړه GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)

بنية ggml_hash_سېټ {
    size_t اندازه;
    ggml_bitset_t * کارول_شوی;       // whether or not the keys are in use i.e. set
    بنية ggml_tensor ** کلیګانې; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};

بنية ggml_hash_سېټ ggml_hash_سېټ_new(size_t اندازه);
فراغ                 ggml_hash_سېټ_free(بنية ggml_hash_سېټ * hash_سېټ);

// returns the minimum size for a hash set that can hold min_sz elements
size_t ggml_hash_اندازه(size_t min_sz);

// remove all elements from the hash set
فراغ ggml_hash_سېټ_reset(بنية ggml_hash_سېټ * hash_سېټ);

// returns true if key is in the hash set
ساكن bool ggml_hash_لري(ثابت بنية ggml_hash_سېټ * hash_سېټ, بنية ggml_tensor * کلی);

// returns GGML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
ساكن size_t ggml_hash_پیدا_کړه(ثابت بنية ggml_hash_سېټ * hash_سېټ, ثابت بنية ggml_tensor * کلی);

// returns GGML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
ساكن size_t ggml_hash_ورننباسه(بنية ggml_hash_سېټ * hash_سېټ, بنية ggml_tensor * کلی);

// return index, asserts if table is full
ساكن size_t ggml_hash_پیدا_کړه_or_ورننباسه(بنية ggml_hash_سېټ * hash_سېټ, بنية ggml_tensor * کلی);

// hash function for ggml_tensor
ساكن مضمن size_t ggml_hash(ثابت بنية ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    أرجع (size_t)(uintptr_t)p >> 4;
}

ساكن size_t ggml_hash_پیدا_کړه(ثابت بنية ggml_hash_سېټ * hash_سېټ, ثابت بنية ggml_tensor * کلی) {
    size_t h = ggml_hash(کلی) % hash_سېټ->اندازه;

    // linear probing
    size_t i = h;
    طالما (ggml_bitset_get(hash_سېټ->کارول_شوی, i) && hash_سېټ->کلیګانې[i] != کلی) {
        i = (i + 1) % hash_سېټ->اندازه;
        إذا (i == h) {
            // visited all hash table entries -> not found
            أرجع GGML_HASHSET_ډک;
        }
    }
    أرجع i;
}

ساكن bool ggml_hash_لري(ثابت بنية ggml_hash_سېټ * hash_سېټ, بنية ggml_tensor * کلی) {
    size_t i = ggml_hash_پیدا_کړه(hash_سېټ, کلی);
    أرجع i != GGML_HASHSET_ډک && ggml_bitset_get(hash_سېټ->کارول_شوی, i);
}

ساكن size_t ggml_hash_ورننباسه(بنية ggml_hash_سېټ * hash_سېټ, بنية ggml_tensor * کلی) {
    size_t h = ggml_hash(کلی) % hash_سېټ->اندازه;

    // linear probing
    size_t i = h;
    افعل {
        إذا (!ggml_bitset_get(hash_سېټ->کارول_شوی, i)) {
            ggml_bitset_سېټ(hash_سېټ->کارول_شوی, i);
            hash_سېټ->کلیګانې[i] = کلی;
            أرجع i;
        }
        إذا (hash_سېټ->کلیګانې[i] == کلی) {
            أرجع GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_سېټ->اندازه;
    } طالما (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

ساكن size_t ggml_hash_پیدا_کړه_or_ورننباسه(بنية ggml_hash_سېټ * hash_سېټ, بنية ggml_tensor * کلی) {
    size_t h = ggml_hash(کلی) % hash_سېټ->اندازه;

    // linear probing
    size_t i = h;
    افعل {
        إذا (!ggml_bitset_get(hash_سېټ->کارول_شوی, i)) {
            ggml_bitset_سېټ(hash_سېټ->کارول_شوی, i);
            hash_سېټ->کلیګانې[i] = کلی;
            أرجع i;
        }
        إذا (hash_سېټ->کلیګانې[i] == کلی) {
            أرجع i;
        }
        i = (i + 1) % hash_سېټ->اندازه;
    } طالما (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

// computation graph

تعداد ggml_cgraph_eval_ترتیب {
    GGML_CGRAPH_EVAL_ترتیب_کیڼ_منزل_ښی = 0,
    GGML_CGRAPH_EVAL_ترتیب_ښی_منزل_کیڼ,
    GGML_CGRAPH_EVAL_ترتیب_شمېر
};

بنية ggml_cgraph {
    صحيح اندازه;    // maximum number of nodes/leafs/grads/grad_accs
    صحيح n_nodes; // number of nodes currently in use
    صحيح n_leafs; // number of leafs currently in use

    بنية ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
    بنية ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
    بنية ggml_tensor ** grad_accs; // accumulators for node gradients
    بنية ggml_tensor ** leafs;     // tensors with constant data
    int32_t             * use_counts;// number of uses of each tensor, indexed by hash table slot

    بنية ggml_hash_سېټ لیدل_شوی_hash_سېټ;

    تعداد ggml_cgraph_eval_ترتیب ترتیب;

    // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
    // a value of 0 means it is not set and should be ignored
    uint64_t uid;
};

// returns a slice of cgraph with nodes [i0, i1)
// the slice does not have leafs or gradients
// if you need the gradients, get them from the original graph
بنية ggml_cgraph ggml_ګراف_view(بنية ggml_cgraph * cgraph, صحيح i0, صحيح i1);

// ggml-alloc.c: true if the operation can reuse memory from its sources
GGML_API bool ggml_op_can_inplace(تعداد ggml_op op);


// Memory allocation

GGML_API فراغ * ggml_aligned_malloc(size_t اندازه);
GGML_API فراغ ggml_aligned_free(فراغ * ptr, size_t اندازه);

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

ساكن مضمن عائم fp32_from_بټونه(uint32_t w) {
    اتحاد {
        uint32_t as_بټونه;
        عائم as_ارزښت;
    } fp32;
    fp32.as_بټونه = w;
    أرجع fp32.as_ارزښت;
}

ساكن مضمن uint32_t fp32_منزل_بټونه(عائم f) {
    اتحاد {
        عائم as_ارزښت;
        uint32_t as_بټونه;
    } fp32;
    fp32.as_ارزښت = f;
    أرجع fp32.as_بټونه;
}

ساكن مضمن عائم ggml_compute_fp16_منزل_fp32(ggml_fp16_t h) {
    ثابت uint32_t w = (uint32_t) h << 16;
    ثابت uint32_t sign = w & UINT32_C(0x80000000);
    ثابت uint32_t two_w = w + w;

    ثابت uint32_t exp_دننه_واټن = UINT32_C(0xE0) << 23;
#إذا (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    ثابت عائم exp_scale = 0x1.0p-112f;
#وإلا
    ثابت عائم exp_scale = fp32_from_بټونه(UINT32_C(0x7800000));
#د_که_پای
    ثابت عائم normalized_ارزښت = fp32_from_بټونه((two_w >> 4) + exp_دننه_واټن) * exp_scale;

    ثابت uint32_t magic_ماسک = UINT32_C(126) << 23;
    ثابت عائم magic_bias = 0.5f;
    ثابت عائم denormalized_ارزښت = fp32_from_بټونه((two_w >> 17) | magic_ماسک) - magic_bias;

    ثابت uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    ثابت uint32_t پایله = sign |
        (two_w < denormalized_cutoff ? fp32_منزل_بټونه(denormalized_ارزښت) : fp32_منزل_بټونه(normalized_ارزښت));
    أرجع fp32_from_بټونه(پایله);
}

ساكن مضمن ggml_fp16_t ggml_compute_fp32_منزل_fp16(عائم f) {
#إذا (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    ثابت عائم scale_منزل_inf = 0x1.0p+112f;
    ثابت عائم scale_منزل_zero = 0x1.0p-110f;
#وإلا
    ثابت عائم scale_منزل_inf = fp32_from_بټونه(UINT32_C(0x77800000));
    ثابت عائم scale_منزل_zero = fp32_from_بټونه(UINT32_C(0x08800000));
#د_که_پای
    عائم بنسټ = (fabsf(f) * scale_منزل_inf) * scale_منزل_zero;

    ثابت uint32_t w = fp32_منزل_بټونه(f);
    ثابت uint32_t shl1_w = w + w;
    ثابت uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    إذا (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    بنسټ = fp32_from_بټونه((bias >> 1) + UINT32_C(0x07800000)) + بنسټ;
    ثابت uint32_t بټونه = fp32_منزل_بټونه(بنسټ);
    ثابت uint32_t exp_بټونه = (بټونه >> 13) & UINT32_C(0x00007C00);
    ثابت uint32_t mantissa_بټونه = بټونه & UINT32_C(0x00000FFF);
    ثابت uint32_t nonsign = exp_بټونه + mantissa_بټونه;
    أرجع (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#تعریف_کړه GGML_COMPUTE_FP16_منزل_FP32(x) ggml_compute_fp16_منزل_fp32(x)
#تعریف_کړه GGML_COMPUTE_FP32_منزل_FP16(x) ggml_compute_fp32_منزل_fp16(x)

#تعریف_کړه GGML_FP16_منزل_FP32(x) GGML_COMPUTE_FP16_منزل_FP32(x)
#تعریف_کړه GGML_FP32_منزل_FP16(x) GGML_COMPUTE_FP32_منزل_FP16(x)

ساكن مضمن عائم ggml_e8m0_منزل_fp32(uint8_t x) {
    uint32_t بټونه;  // Stores the raw bit representation of the float

    // Handle special case for minimum exponent (denormalized float)
    إذا (x == 0) {
        // Bit pattern for 2^(-127):
        // - Sign bit: 0 (positive)
        // - Exponent: 0 (denormalized number)
        // - Mantissa: 0x400000 (0.5 in fractional form)
        // Value = 0.5 * 2^(-126) = 2^(-127)
        بټونه = 0x00400000;
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
        بټونه = (uint32_t) x << 23;
    }

    عائم پایله;  // Final float value
                   // Safely reinterpret bit pattern as float without type-punning issues
    حافظه_نقل(&پایله, &بټونه, حجم(عائم));
    أرجع پایله;
}

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
ساكن مضمن عائم ggml_e8m0_منزل_fp32_half(uint8_t x) {
    uint32_t بټونه;

    // For x < 2: use precomputed denormal patterns
    إذا (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        بټونه = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    وإلا {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        بټونه = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    عائم پایله;
    حافظه_نقل(&پایله, &بټونه, حجم(عائم));
    أرجع پایله;
}

#تعریف_کړه GGML_E8_M0_منزل_FP32(x) ggml_e8m0_منزل_fp32(x)
#تعریف_کړه GGML_E8_M0_منزل_FP32_HALF(x) ggml_e8m0_منزل_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
ساكن مضمن عائم ggml_ue4m3_منزل_fp32(uint8_t x) {
    إذا (x == 0 || x == 0x7F) {
        أرجع 0.0f;
    }
    صحيح   توانیز_تابع = (x >> 3) & 0xF;
    صحيح   man = x & 0x7;
    عائم raw;
    إذا (توانیز_تابع == 0) {
        raw = ldexpf((عائم) man, -9);
    } وإلا {
        raw = ldexpf(1.0f + (عائم) man / 8.0f, توانیز_تابع - 7);
    }
    أرجع raw * 0.5f;
}

ساكن مضمن uint8_t ggml_fp32_منزل_ue4m3(عائم x) {
    إذا (!(x > 0.0f)) {
        أرجع 0;
    }
    إذا (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t بټونه;
    حافظه_نقل(&بټونه, &x, 4);
    صحيح fp32_exp  = ((بټونه >> 23) & 0xFF) - 127;
    صحيح fp32_man  = (بټونه >> 20) & 0x7;
    صحيح ue4m3_exp = fp32_exp + 7;
    إذا (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        صحيح man = (صحيح) (x * 512.0f + 0.5f);
        إذا (man > 7) {
            man = 7;
        }
        إذا (man < 1) {
            أرجع 0;
        }
        أرجع (uint8_t) man;
    }
    إذا (ue4m3_exp >= 15) {
        أرجع 0x7E;
    }
    صحيح پړاو_بټ = (بټونه >> 19) & 1;
    صحيح ue4m3_man = fp32_man + پړاو_بټ;
    إذا (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        إذا (ue4m3_exp >= 15) {
            أرجع 0x7E;
        }
    }
    أرجع (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
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
ساكن مضمن عائم ggml_compute_bf16_منزل_fp32(ggml_bf16_t h) {
    اتحاد {
        عائم f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.بټونه << 16;
    أرجع u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
ساكن مضمن ggml_bf16_t ggml_compute_fp32_منزل_bf16(عائم s) {
    ggml_bf16_t h;
    اتحاد {
        عائم f;
        uint32_t i;
    } u;
    u.f = s;
    إذا ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.بټونه = (u.i >> 16) | 64; /* force to quiet */
        أرجع h;
    }
    h.بټونه = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    أرجع h;
}

#تعریف_کړه GGML_FP32_منزل_BF16(x) ggml_compute_fp32_منزل_bf16(x)
#تعریف_کړه GGML_BF16_منزل_FP32(x) ggml_compute_bf16_منزل_fp32(x)

ساكن مضمن int32_t ggml_غوټه_get_use_شمېر(ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx) {
    ثابت بنية ggml_tensor * غوټه = cgraph->nodes[غوټه_idx];

    size_t hash_pos = ggml_hash_پیدا_کړه(&cgraph->لیدل_شوی_hash_سېټ, غوټه);
    إذا (!ggml_bitset_get(cgraph->لیدل_شوی_hash_سېټ.کارول_شوی, hash_pos)) {
        أرجع 0;
    }
    أرجع cgraph->use_counts[hash_pos];
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
ساكن مضمن bool ggml_غوټه_has_n_کاروي(ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx, int32_t n_کاروي) {
    ثابت بنية ggml_tensor * غوټه = cgraph->nodes[غوټه_idx];

    // check the use count against how many we're replacing
    إذا (ggml_غوټه_get_use_شمېر(cgraph, غوټه_idx) != n_کاروي) {
        أرجع false;
    }

    // if node is a view, some other node might be using the intermediate result
    // via the view source.
    إذا (غوټه->view_src) {
        أرجع false;
    }

    // If the user requested output for the node, can't fuse
    إذا (غوټه->flags & GGML_TENSOR_FLAG_OUTPUT) {
        أرجع false;
    }

    أرجع true;
}

// Returns true if nodes with indices { node_idxs } are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
ساكن مضمن bool ggml_can_fuse_ext(ثابت بنية ggml_cgraph * cgraph, ثابت صحيح * غوټه_idxs, ثابت تعداد ggml_op * ops, صحيح عدد_ops) {
    لكل (صحيح i = 0; i < عدد_ops; ++i) {
        إذا (غوټه_idxs[i] >= cgraph->n_nodes) {
            أرجع false;
        }

        بنية ggml_tensor * غوټه = cgraph->nodes[غوټه_idxs[i]];
        إذا (غوټه->op != ops[i]) {
            أرجع false;
        }
        إذا ((غوټه->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            أرجع false;
        }
        إذا (i < عدد_ops - 1 && !ggml_غوټه_has_n_کاروي(cgraph, غوټه_idxs[i], 1)) {
            أرجع false;
        }
        إذا (i > 0) {
            بنية ggml_tensor * prev = cgraph->nodes[غوټه_idxs[i - 1]];
            إذا (غوټه->src[0] != prev && غوټه->src[1] != prev) {
                أرجع false;
            }
            إذا (!ggml_are_same_shape(غوټه, prev)) {
                أرجع false;
            }
        }
    }
    أرجع true;
}

// same as above, for sequential indices starting at node_idx
ساكن مضمن bool ggml_can_fuse(ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx, ثابت تعداد ggml_op * ops, صحيح عدد_ops) {
    أكد(عدد_ops < 32);

    إذا (غوټه_idx + عدد_ops > cgraph->n_nodes) {
        أرجع false;
    }

    صحيح idxs[32];
    لكل (صحيح i = 0; i < عدد_ops; ++i) {
        idxs[i] = غوټه_idx + i;
    }

    أرجع ggml_can_fuse_ext(cgraph, idxs, ops, عدد_ops);
}

GGML_API bool ggml_can_fuse_subgraph_ext(ثابت بنية ggml_cgraph * cgraph,
                                         ثابت صحيح *                غوټه_idxs,
                                         صحيح                        شمېر,
                                         ثابت تعداد ggml_op *       ops,
                                         ثابت صحيح *                outputs,
                                         صحيح                        عدد_outputs);

// Returns true if the subgraph formed by {node_idxs} can be fused
// checks whethers all nodes which are not part of outputs can be elided
// by checking if their num_uses are confined to the subgraph
ساكن مضمن bool ggml_can_fuse_subgraph(ثابت بنية ggml_cgraph * cgraph,
                                          صحيح                        غوټه_idx,
                                          صحيح                        شمېر,
                                          ثابت تعداد ggml_op *       ops,
                                          ثابت صحيح *                outputs,
                                          صحيح                        عدد_outputs) {
    GGML_ASSERT(شمېر < 32);
    إذا (غوټه_idx + شمېر > cgraph->n_nodes) {
        أرجع false;
    }

    صحيح idxs[32];

    لكل (صحيح i = 0; i < شمېر; ++i) {
        idxs[i] = غوټه_idx + i;
    }

    أرجع ggml_can_fuse_subgraph_ext(cgraph, idxs, شمېر, ops, outputs, عدد_outputs);
}

#که_تعریف_شوی __cplusplus
}
#د_که_پای

#که_تعریف_شوی __cplusplus
#ضمن <array>
#ضمن <initializer_list>
#ضمن <vector>

// nicer C++ syntax for ggml_can_fuse
مضمن bool ggml_can_fuse(ثابت بنية ggml_cgraph * cgraph, صحيح غوټه_idx, std::initializer_list<تعداد ggml_op> ops) {
    أرجع ggml_can_fuse(cgraph, غوټه_idx, ops.begin(), (صحيح)ops.اندازه());
}

مضمن bool ggml_can_fuse_subgraph(ثابت بنية ggml_cgraph *          cgraph,
                                   صحيح                                 پیل_idx,
                                   std::initializer_list<تعداد ggml_op> ops,
                                   std::initializer_list<صحيح>          outputs = {}) {
    أرجع ggml_can_fuse_subgraph(cgraph, پیل_idx, ops.اندازه(), ops.begin(), outputs.begin(), outputs.اندازه());
}

// Return true if the edges in the graph match expectations.
مضمن bool ggml_check_څنډې(ثابت بنية ggml_cgraph *                cgraph,
                             صحيح                                       پیل_idx,
                             std::initializer_list<std::array<صحيح, 3>> څنډې) {
    لكل (ثابت خپلکاری & څنډه : څنډې) {
        صحيح dst_غوټه = څنډه[0];
        صحيح src_idx  = څنډه[1];
        صحيح src_غوټه = څنډه[2];
        إذا (cgraph->nodes[پیل_idx + dst_غوټه]->src[src_idx] != cgraph->nodes[پیل_idx + src_غوټه]) {
            أرجع false;
        }
    }
    أرجع true;
}

// expose GGUF internals for test code
GGML_API size_t gguf_type_اندازه(تعداد gguf_type type);
GGML_API فراغ gguf_write_منزل_buf(ثابت بنية gguf_context * ctx, std::ویکټور<int8_t> & buf, bool only_meta);
#د_که_پای // __cplusplus
