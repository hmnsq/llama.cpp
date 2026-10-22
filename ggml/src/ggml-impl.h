#ਹਦਾਇਤ once

// GGML internal header

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "gguf.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __ARM_FEATURE_SVE
#ਅਨ੍ਤਰ੍ਭਾਵ <arm_sve.h>
#ਜੇ_ਅੰਤ // __ARM_FEATURE_SVE

#ਜੇ defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#ਅਨ੍ਤਰ੍ਭਾਵ <arm_neon.h>
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ

ਖਾਲੀ ggml_print_backtrace(ਖਾਲੀ);

uint64_t ggml_ਗ੍ਰਾਫ_ਅਗਲਾ_uid(ਖਾਲੀ);

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ MIN
#    define MIN(a, b) ((a) < (b) ? (a) : (b))
#ਜੇ_ਅੰਤ

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ MAX
#    define MAX(a, b) ((a) > (b) ? (a) : (b))
#ਜੇ_ਅੰਤ

// required for mmap as gguf only guarantees 32-byte alignment
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TENSOR_ALIGNMENT 32

// static_assert should be a #define, but if it's not,
// fall back to the _Static_assert C11 keyword.
// if C99 - static_assert is noop
// ref: https://stackoverflow.com/a/53923785/4039976
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ __cplusplus
    #ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ static_assert
        #ਜੇ defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
            #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ static_assert(cond, ਸੁਨੇਹਾ_2) _Static_assert(cond, ਸੁਨੇਹਾ_2)
        #ਨਹੀਂਤਾਂ
            #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ static_assert(cond, ਸੁਨੇਹਾ_2) ਬਣਤਰ global_scope_noop_trick
        #ਜੇ_ਅੰਤ
    #ਜੇ_ਅੰਤ
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ ggml_up32(ਪੂਰਨ n) {
    ਵਾਪਸ (n + 31) & ~31;
}

//static inline int ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ ggml_ਉੱਪਰ(ਪੂਰਨ n, ਪੂਰਨ m) {
    // assert m is a power of 2
    GGML_ASSERT((m & (m - 1)) == 0);
    ਵਾਪਸ (n + m - 1) & ~(m - 1);
}

// TODO: move to ggml.h? (won't be able to inline)
ਸਥਿਰ bool ggml_are_same_layout(ਅਚਲ ਬਣਤਰ ggml_tensor * a, ਅਚਲ ਬਣਤਰ ggml_tensor * b) {
    ਜੇ (a->type != b->type) {
        ਵਾਪਸ false;
    }
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; i++) {
        ਜੇ (a->ne[i] != b->ne[i]) {
            ਵਾਪਸ false;
        }
        ਜੇ (a->nb[i] != b->nb[i]) {
            ਵਾਪਸ false;
        }
    }
    ਵਾਪਸ true;
}

ਸਥਿਰ bool ggml_op_is_ਖਾਲੀ_ਥਾਵਾਂ(ਗਿਣਤੀ ggml_op op) {
    ਚੋਣ (op) {
        ਹਾਲਤ GGML_OP_NONE:
        ਹਾਲਤ GGML_OP_RESHAPE:
        ਹਾਲਤ GGML_OP_TRANSPOSE:
        ਹਾਲਤ GGML_OP_VIEW:
        ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ:
            ਵਾਪਸ true;
        ਮੂਲ:
            ਵਾਪਸ false;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool ggml_impl_is_view(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    ਵਾਪਸ t->view_src != ਨਿਰਾ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ ggml_compute_softplus_f32(ਦਸ਼ਮਲਵ ਨਿਵੇਸ਼) {
    ਵਾਪਸ (ਨਿਵੇਸ਼ > 20.0f) ? ਨਿਵੇਸ਼ : logf(1 + expf(ਨਿਵੇਸ਼));
}
//
// logging
//

GGML_ATTRIBUTE_FORMAT(2, 3)
GGML_API ਖਾਲੀ ggml_log_ਅੰਦਰੂਨੀ        (ਗਿਣਤੀ ggml_log_level level, ਅਚਲ ਅੱਖਰ * format, ...);
GGML_API ਖਾਲੀ ggml_log_callback_default(ਗਿਣਤੀ ggml_log_level level, ਅਚਲ ਅੱਖਰ * ਪਾਠ_2, ਖਾਲੀ * user_ਅੰਕੜੇ);

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_LOG(...)       ggml_log_ਅੰਦਰੂਨੀ(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_LOG_INFO(...)  ggml_log_ਅੰਦਰੂਨੀ(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_LOG_WARN(...)  ggml_log_ਅੰਦਰੂਨੀ(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_LOG_ERROR(...) ggml_log_ਅੰਦਰੂਨੀ(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_LOG_DEBUG(...) ggml_log_ਅੰਦਰੂਨੀ(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_LOG_CONT(...)  ggml_log_ਅੰਦਰੂਨੀ(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_DEBUG 0

#ਜੇ (GGML_DEBUG >= 1)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_PRINT_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_PRINT_DEBUG(...)
#ਜੇ_ਅੰਤ

#ਜੇ (GGML_DEBUG >= 5)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_PRINT_DEBUG_5(...) GGML_LOG_DEBUG(__VA_ARGS__)
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_PRINT_DEBUG_5(...)
#ਜੇ_ਅੰਤ

#ਜੇ (GGML_DEBUG >= 10)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_PRINT_DEBUG_10(...) GGML_LOG_DEBUG(__VA_ARGS__)
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_PRINT_DEBUG_10(...)
#ਜੇ_ਅੰਤ

// tensor params

ਸਥਿਰ ਖਾਲੀ ggml_ਸੈੱਟ_op_params(ਬਣਤਰ ggml_tensor * tensor, ਅਚਲ ਖਾਲੀ * params, size_t params_ਮਾਪ) {
    GGML_ASSERT(tensor != ਨਿਰਾ); // silence -Warray-bounds warnings
    ਪ੍ਰਤਿਜ੍ਞਾ(params_ਮਾਪ <= GGML_MAX_OP_PARAMS);
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(tensor->op_params, params, params_ਮਾਪ);
}

ਸਥਿਰ int32_t ggml_get_op_params_i32(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, uint32_t i) {
    ਪ੍ਰਤਿਜ੍ਞਾ(i < GGML_MAX_OP_PARAMS / ਆਕਾਰ(int32_t));
    ਵਾਪਸ ((ਅਚਲ int32_t *)(tensor->op_params))[i];
}

ਸਥਿਰ ਦਸ਼ਮਲਵ ggml_get_op_params_f32(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, uint32_t i) {
    ਪ੍ਰਤਿਜ੍ਞਾ(i < GGML_MAX_OP_PARAMS / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਵਾਪਸ ((ਅਚਲ ਦਸ਼ਮਲਵ *)(tensor->op_params))[i];
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
ਸਥਿਰ ਖਾਲੀ ggml_ਸੈੱਟ_op_params_i32(ਬਣਤਰ ggml_tensor * tensor, uint32_t i, int32_t ਮੁੱਲ) {
    ਪ੍ਰਤਿਜ੍ਞਾ(i < GGML_MAX_OP_PARAMS / ਆਕਾਰ(int32_t));
    ((int32_t *)(tensor->op_params))[i] = ਮੁੱਲ;
}

ਸਥਿਰ ਖਾਲੀ ggml_ਸੈੱਟ_op_params_f32(ਬਣਤਰ ggml_tensor * tensor, uint32_t i, ਦਸ਼ਮਲਵ ਮੁੱਲ) {
    ਪ੍ਰਤਿਜ੍ਞਾ(i < GGML_MAX_OP_PARAMS / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ((ਦਸ਼ਮਲਵ *)(tensor->op_params))[i] = ਮੁੱਲ;
}

ਬਣਤਰ ggml_map_custom1_op_params {
    ggml_custom1_op_t  fun;
    ਪੂਰਨ                n_tasks;
    ਖਾਲੀ             * userdata;
};

ਬਣਤਰ ggml_map_custom2_op_params {
    ggml_custom2_op_t   fun;
    ਪੂਰਨ                 n_tasks;
    ਖਾਲੀ              * userdata;
};

ਬਣਤਰ ggml_map_custom3_op_params {
    ggml_custom3_op_t fun;
    ਪੂਰਨ               n_tasks;
    ਖਾਲੀ            * userdata;
};

ਬਣਤਰ ggml_custom_op_params {
    ggml_custom_op_t fun;
    ਪੂਰਨ              n_tasks;
    ਖਾਲੀ           * userdata;
};

// bitset

ਕਿਸਮਪਰਿਭਾਸ਼ਾ uint32_t ggml_bitset_t;

static_assert(ਆਕਾਰ(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ BITSET_ਨਕਾਬ (ਆਕਾਰ(ggml_bitset_t)*8 - 1)

ਸਥਿਰ size_t ggml_bitset_ਮਾਪ(size_t n) {
    ਵਾਪਸ (n + BITSET_ਨਕਾਬ) >> BITSET_SHR;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool ggml_bitset_get(ਅਚਲ ggml_bitset_t * bitset, size_t i) {
    ਵਾਪਸ !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_ਨਕਾਬ)));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_bitset_ਸੈੱਟ(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_ਨਕਾਬ));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_bitset_clear(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_ਨਕਾਬ));
}

// hash set

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_HASHSET_ਭਰਿਆ ((size_t)-1)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)

ਬਣਤਰ ggml_hash_ਸੈੱਟ {
    size_t ਮਾਪ;
    ggml_bitset_t * ਵਰਤਿਆ;       // whether or not the keys are in use i.e. set
    ਬਣਤਰ ggml_tensor ** ਕੁੰਜੀਆਂ; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};

ਬਣਤਰ ggml_hash_ਸੈੱਟ ggml_hash_ਸੈੱਟ_new(size_t ਮਾਪ);
ਖਾਲੀ                 ggml_hash_ਸੈੱਟ_free(ਬਣਤਰ ggml_hash_ਸੈੱਟ * hash_ਸੈੱਟ);

// returns the minimum size for a hash set that can hold min_sz elements
size_t ggml_hash_ਮਾਪ(size_t min_sz);

// remove all elements from the hash set
ਖਾਲੀ ggml_hash_ਸੈੱਟ_reset(ਬਣਤਰ ggml_hash_ਸੈੱਟ * hash_ਸੈੱਟ);

// returns true if key is in the hash set
ਸਥਿਰ bool ggml_hash_ਸ਼ਾਮਲ_ਹੈ(ਅਚਲ ਬਣਤਰ ggml_hash_ਸੈੱਟ * hash_ਸੈੱਟ, ਬਣਤਰ ggml_tensor * ਕੁੰਜੀ);

// returns GGML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
ਸਥਿਰ size_t ggml_hash_ਲੱਭੋ(ਅਚਲ ਬਣਤਰ ggml_hash_ਸੈੱਟ * hash_ਸੈੱਟ, ਅਚਲ ਬਣਤਰ ggml_tensor * ਕੁੰਜੀ);

// returns GGML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
ਸਥਿਰ size_t ggml_hash_ਪਾਓ(ਬਣਤਰ ggml_hash_ਸੈੱਟ * hash_ਸੈੱਟ, ਬਣਤਰ ggml_tensor * ਕੁੰਜੀ);

// return index, asserts if table is full
ਸਥਿਰ size_t ggml_hash_ਲੱਭੋ_or_ਪਾਓ(ਬਣਤਰ ggml_hash_ਸੈੱਟ * hash_ਸੈੱਟ, ਬਣਤਰ ggml_tensor * ਕੁੰਜੀ);

// hash function for ggml_tensor
ਸਥਿਰ ਸਤਰਅੰਦਰ size_t ggml_hash(ਅਚਲ ਬਣਤਰ ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    ਵਾਪਸ (size_t)(uintptr_t)p >> 4;
}

ਸਥਿਰ size_t ggml_hash_ਲੱਭੋ(ਅਚਲ ਬਣਤਰ ggml_hash_ਸੈੱਟ * hash_ਸੈੱਟ, ਅਚਲ ਬਣਤਰ ggml_tensor * ਕੁੰਜੀ) {
    size_t h = ggml_hash(ਕੁੰਜੀ) % hash_ਸੈੱਟ->ਮਾਪ;

    // linear probing
    size_t i = h;
    ਜਦੋਂ (ggml_bitset_get(hash_ਸੈੱਟ->ਵਰਤਿਆ, i) && hash_ਸੈੱਟ->ਕੁੰਜੀਆਂ[i] != ਕੁੰਜੀ) {
        i = (i + 1) % hash_ਸੈੱਟ->ਮਾਪ;
        ਜੇ (i == h) {
            // visited all hash table entries -> not found
            ਵਾਪਸ GGML_HASHSET_ਭਰਿਆ;
        }
    }
    ਵਾਪਸ i;
}

ਸਥਿਰ bool ggml_hash_ਸ਼ਾਮਲ_ਹੈ(ਅਚਲ ਬਣਤਰ ggml_hash_ਸੈੱਟ * hash_ਸੈੱਟ, ਬਣਤਰ ggml_tensor * ਕੁੰਜੀ) {
    size_t i = ggml_hash_ਲੱਭੋ(hash_ਸੈੱਟ, ਕੁੰਜੀ);
    ਵਾਪਸ i != GGML_HASHSET_ਭਰਿਆ && ggml_bitset_get(hash_ਸੈੱਟ->ਵਰਤਿਆ, i);
}

ਸਥਿਰ size_t ggml_hash_ਪਾਓ(ਬਣਤਰ ggml_hash_ਸੈੱਟ * hash_ਸੈੱਟ, ਬਣਤਰ ggml_tensor * ਕੁੰਜੀ) {
    size_t h = ggml_hash(ਕੁੰਜੀ) % hash_ਸੈੱਟ->ਮਾਪ;

    // linear probing
    size_t i = h;
    ਕਰੋ {
        ਜੇ (!ggml_bitset_get(hash_ਸੈੱਟ->ਵਰਤਿਆ, i)) {
            ggml_bitset_ਸੈੱਟ(hash_ਸੈੱਟ->ਵਰਤਿਆ, i);
            hash_ਸੈੱਟ->ਕੁੰਜੀਆਂ[i] = ਕੁੰਜੀ;
            ਵਾਪਸ i;
        }
        ਜੇ (hash_ਸੈੱਟ->ਕੁੰਜੀਆਂ[i] == ਕੁੰਜੀ) {
            ਵਾਪਸ GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_ਸੈੱਟ->ਮਾਪ;
    } ਜਦੋਂ (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

ਸਥਿਰ size_t ggml_hash_ਲੱਭੋ_or_ਪਾਓ(ਬਣਤਰ ggml_hash_ਸੈੱਟ * hash_ਸੈੱਟ, ਬਣਤਰ ggml_tensor * ਕੁੰਜੀ) {
    size_t h = ggml_hash(ਕੁੰਜੀ) % hash_ਸੈੱਟ->ਮਾਪ;

    // linear probing
    size_t i = h;
    ਕਰੋ {
        ਜੇ (!ggml_bitset_get(hash_ਸੈੱਟ->ਵਰਤਿਆ, i)) {
            ggml_bitset_ਸੈੱਟ(hash_ਸੈੱਟ->ਵਰਤਿਆ, i);
            hash_ਸੈੱਟ->ਕੁੰਜੀਆਂ[i] = ਕੁੰਜੀ;
            ਵਾਪਸ i;
        }
        ਜੇ (hash_ਸੈੱਟ->ਕੁੰਜੀਆਂ[i] == ਕੁੰਜੀ) {
            ਵਾਪਸ i;
        }
        i = (i + 1) % hash_ਸੈੱਟ->ਮਾਪ;
    } ਜਦੋਂ (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

// computation graph

ਗਿਣਤੀ ggml_cgraph_eval_ਕ੍ਰਮ {
    GGML_CGRAPH_EVAL_ਕ੍ਰਮ_ਖੱਬਾ_ਮੰਜ਼ਿਲ_ਸੱਜਾ = 0,
    GGML_CGRAPH_EVAL_ਕ੍ਰਮ_ਸੱਜਾ_ਮੰਜ਼ਿਲ_ਖੱਬਾ,
    GGML_CGRAPH_EVAL_ਕ੍ਰਮ_ਗਿਣਤੀ
};

ਬਣਤਰ ggml_cgraph {
    ਪੂਰਨ ਮਾਪ;    // maximum number of nodes/leafs/grads/grad_accs
    ਪੂਰਨ n_nodes; // number of nodes currently in use
    ਪੂਰਨ n_leafs; // number of leafs currently in use

    ਬਣਤਰ ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
    ਬਣਤਰ ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
    ਬਣਤਰ ggml_tensor ** grad_accs; // accumulators for node gradients
    ਬਣਤਰ ggml_tensor ** leafs;     // tensors with constant data
    int32_t             * use_counts;// number of uses of each tensor, indexed by hash table slot

    ਬਣਤਰ ggml_hash_ਸੈੱਟ ਦੇਖਿਆ_hash_ਸੈੱਟ;

    ਗਿਣਤੀ ggml_cgraph_eval_ਕ੍ਰਮ ਕ੍ਰਮ;

    // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
    // a value of 0 means it is not set and should be ignored
    uint64_t uid;
};

// returns a slice of cgraph with nodes [i0, i1)
// the slice does not have leafs or gradients
// if you need the gradients, get them from the original graph
ਬਣਤਰ ggml_cgraph ggml_ਗ੍ਰਾਫ_view(ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ i0, ਪੂਰਨ i1);

// ggml-alloc.c: true if the operation can reuse memory from its sources
GGML_API bool ggml_op_can_inplace(ਗਿਣਤੀ ggml_op op);


// Memory allocation

GGML_API ਖਾਲੀ * ggml_aligned_malloc(size_t ਮਾਪ);
GGML_API ਖਾਲੀ ggml_aligned_free(ਖਾਲੀ * ptr, size_t ਮਾਪ);

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ fp32_from_ਬਿੱਟ(uint32_t w) {
    ਸੰਘ {
        uint32_t as_ਬਿੱਟ;
        ਦਸ਼ਮਲਵ as_ਮੁੱਲ;
    } fp32;
    fp32.as_ਬਿੱਟ = w;
    ਵਾਪਸ fp32.as_ਮੁੱਲ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t fp32_ਮੰਜ਼ਿਲ_ਬਿੱਟ(ਦਸ਼ਮਲਵ f) {
    ਸੰਘ {
        ਦਸ਼ਮਲਵ as_ਮੁੱਲ;
        uint32_t as_ਬਿੱਟ;
    } fp32;
    fp32.as_ਮੁੱਲ = f;
    ਵਾਪਸ fp32.as_ਬਿੱਟ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ ggml_compute_fp16_ਮੰਜ਼ਿਲ_fp32(ggml_fp16_t h) {
    ਅਚਲ uint32_t w = (uint32_t) h << 16;
    ਅਚਲ uint32_t sign = w & UINT32_C(0x80000000);
    ਅਚਲ uint32_t two_w = w + w;

    ਅਚਲ uint32_t exp_ਵਿੱਥ = UINT32_C(0xE0) << 23;
#ਜੇ (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    ਅਚਲ ਦਸ਼ਮਲਵ exp_scale = 0x1.0p-112f;
#ਨਹੀਂਤਾਂ
    ਅਚਲ ਦਸ਼ਮਲਵ exp_scale = fp32_from_ਬਿੱਟ(UINT32_C(0x7800000));
#ਜੇ_ਅੰਤ
    ਅਚਲ ਦਸ਼ਮਲਵ normalized_ਮੁੱਲ = fp32_from_ਬਿੱਟ((two_w >> 4) + exp_ਵਿੱਥ) * exp_scale;

    ਅਚਲ uint32_t magic_ਨਕਾਬ = UINT32_C(126) << 23;
    ਅਚਲ ਦਸ਼ਮਲਵ magic_bias = 0.5f;
    ਅਚਲ ਦਸ਼ਮਲਵ denormalized_ਮੁੱਲ = fp32_from_ਬਿੱਟ((two_w >> 17) | magic_ਨਕਾਬ) - magic_bias;

    ਅਚਲ uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    ਅਚਲ uint32_t ਨਤੀਜਾ = sign |
        (two_w < denormalized_cutoff ? fp32_ਮੰਜ਼ਿਲ_ਬਿੱਟ(denormalized_ਮੁੱਲ) : fp32_ਮੰਜ਼ਿਲ_ਬਿੱਟ(normalized_ਮੁੱਲ));
    ਵਾਪਸ fp32_from_ਬਿੱਟ(ਨਤੀਜਾ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ggml_fp16_t ggml_compute_fp32_ਮੰਜ਼ਿਲ_fp16(ਦਸ਼ਮਲਵ f) {
#ਜੇ (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    ਅਚਲ ਦਸ਼ਮਲਵ scale_ਮੰਜ਼ਿਲ_inf = 0x1.0p+112f;
    ਅਚਲ ਦਸ਼ਮਲਵ scale_ਮੰਜ਼ਿਲ_zero = 0x1.0p-110f;
#ਨਹੀਂਤਾਂ
    ਅਚਲ ਦਸ਼ਮਲਵ scale_ਮੰਜ਼ਿਲ_inf = fp32_from_ਬਿੱਟ(UINT32_C(0x77800000));
    ਅਚਲ ਦਸ਼ਮਲਵ scale_ਮੰਜ਼ਿਲ_zero = fp32_from_ਬਿੱਟ(UINT32_C(0x08800000));
#ਜੇ_ਅੰਤ
    ਦਸ਼ਮਲਵ ਅਧਾਰ = (fabsf(f) * scale_ਮੰਜ਼ਿਲ_inf) * scale_ਮੰਜ਼ਿਲ_zero;

    ਅਚਲ uint32_t w = fp32_ਮੰਜ਼ਿਲ_ਬਿੱਟ(f);
    ਅਚਲ uint32_t shl1_w = w + w;
    ਅਚਲ uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    ਜੇ (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    ਅਧਾਰ = fp32_from_ਬਿੱਟ((bias >> 1) + UINT32_C(0x07800000)) + ਅਧਾਰ;
    ਅਚਲ uint32_t ਬਿੱਟ = fp32_ਮੰਜ਼ਿਲ_ਬਿੱਟ(ਅਧਾਰ);
    ਅਚਲ uint32_t exp_ਬਿੱਟ = (ਬਿੱਟ >> 13) & UINT32_C(0x00007C00);
    ਅਚਲ uint32_t mantissa_ਬਿੱਟ = ਬਿੱਟ & UINT32_C(0x00000FFF);
    ਅਚਲ uint32_t nonsign = exp_ਬਿੱਟ + mantissa_ਬਿੱਟ;
    ਵਾਪਸ (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMPUTE_FP16_ਮੰਜ਼ਿਲ_FP32(x) ggml_compute_fp16_ਮੰਜ਼ਿਲ_fp32(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMPUTE_FP32_ਮੰਜ਼ਿਲ_FP16(x) ggml_compute_fp32_ਮੰਜ਼ਿਲ_fp16(x)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_FP16_ਮੰਜ਼ਿਲ_FP32(x) GGML_COMPUTE_FP16_ਮੰਜ਼ਿਲ_FP32(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_FP32_ਮੰਜ਼ਿਲ_FP16(x) GGML_COMPUTE_FP32_ਮੰਜ਼ਿਲ_FP16(x)

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ ggml_e8m0_ਮੰਜ਼ਿਲ_fp32(uint8_t x) {
    uint32_t ਬਿੱਟ;  // Stores the raw bit representation of the float

    // Handle special case for minimum exponent (denormalized float)
    ਜੇ (x == 0) {
        // Bit pattern for 2^(-127):
        // - Sign bit: 0 (positive)
        // - Exponent: 0 (denormalized number)
        // - Mantissa: 0x400000 (0.5 in fractional form)
        // Value = 0.5 * 2^(-126) = 2^(-127)
        ਬਿੱਟ = 0x00400000;
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
    ਨਹੀਂਤਾਂ {
        // Construct normalized float by shifting exponent into position:
        // - Exponent field: 8 bits (positions 30-23)
        // - Mantissa: 0 (implicit leading 1)
        // Value = 2^(x - 127)
        ਬਿੱਟ = (uint32_t) x << 23;
    }

    ਦਸ਼ਮਲਵ ਨਤੀਜਾ;  // Final float value
                   // Safely reinterpret bit pattern as float without type-punning issues
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਨਤੀਜਾ, &ਬਿੱਟ, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਵਾਪਸ ਨਤੀਜਾ;
}

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ ggml_e8m0_ਮੰਜ਼ਿਲ_fp32_half(uint8_t x) {
    uint32_t ਬਿੱਟ;

    // For x < 2: use precomputed denormal patterns
    ਜੇ (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        ਬਿੱਟ = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    ਨਹੀਂਤਾਂ {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        ਬਿੱਟ = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    ਦਸ਼ਮਲਵ ਨਤੀਜਾ;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਨਤੀਜਾ, &ਬਿੱਟ, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਵਾਪਸ ਨਤੀਜਾ;
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_E8_M0_ਮੰਜ਼ਿਲ_FP32(x) ggml_e8m0_ਮੰਜ਼ਿਲ_fp32(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_E8_M0_ਮੰਜ਼ਿਲ_FP32_HALF(x) ggml_e8m0_ਮੰਜ਼ਿਲ_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ ggml_ue4m3_ਮੰਜ਼ਿਲ_fp32(uint8_t x) {
    ਜੇ (x == 0 || x == 0x7F) {
        ਵਾਪਸ 0.0f;
    }
    ਪੂਰਨ   ਘਾਤਾਂਕ = (x >> 3) & 0xF;
    ਪੂਰਨ   man = x & 0x7;
    ਦਸ਼ਮਲਵ raw;
    ਜੇ (ਘਾਤਾਂਕ == 0) {
        raw = ldexpf((ਦਸ਼ਮਲਵ) man, -9);
    } ਨਹੀਂਤਾਂ {
        raw = ldexpf(1.0f + (ਦਸ਼ਮਲਵ) man / 8.0f, ਘਾਤਾਂਕ - 7);
    }
    ਵਾਪਸ raw * 0.5f;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint8_t ggml_fp32_ਮੰਜ਼ਿਲ_ue4m3(ਦਸ਼ਮਲਵ x) {
    ਜੇ (!(x > 0.0f)) {
        ਵਾਪਸ 0;
    }
    ਜੇ (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t ਬਿੱਟ;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਬਿੱਟ, &x, 4);
    ਪੂਰਨ fp32_exp  = ((ਬਿੱਟ >> 23) & 0xFF) - 127;
    ਪੂਰਨ fp32_man  = (ਬਿੱਟ >> 20) & 0x7;
    ਪੂਰਨ ue4m3_exp = fp32_exp + 7;
    ਜੇ (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        ਪੂਰਨ man = (ਪੂਰਨ) (x * 512.0f + 0.5f);
        ਜੇ (man > 7) {
            man = 7;
        }
        ਜੇ (man < 1) {
            ਵਾਪਸ 0;
        }
        ਵਾਪਸ (uint8_t) man;
    }
    ਜੇ (ue4m3_exp >= 15) {
        ਵਾਪਸ 0x7E;
    }
    ਪੂਰਨ ਗੇੜਾ_ਦੋਆਧਾਰੀ_ਅੰਕ = (ਬਿੱਟ >> 19) & 1;
    ਪੂਰਨ ue4m3_man = fp32_man + ਗੇੜਾ_ਦੋਆਧਾਰੀ_ਅੰਕ;
    ਜੇ (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        ਜੇ (ue4m3_exp >= 15) {
            ਵਾਪਸ 0x7E;
        }
    }
    ਵਾਪਸ (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
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
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ ggml_compute_bf16_ਮੰਜ਼ਿਲ_fp32(ggml_bf16_t h) {
    ਸੰਘ {
        ਦਸ਼ਮਲਵ f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.ਬਿੱਟ << 16;
    ਵਾਪਸ u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
ਸਥਿਰ ਸਤਰਅੰਦਰ ggml_bf16_t ggml_compute_fp32_ਮੰਜ਼ਿਲ_bf16(ਦਸ਼ਮਲਵ s) {
    ggml_bf16_t h;
    ਸੰਘ {
        ਦਸ਼ਮਲਵ f;
        uint32_t i;
    } u;
    u.f = s;
    ਜੇ ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.ਬਿੱਟ = (u.i >> 16) | 64; /* force to quiet */
        ਵਾਪਸ h;
    }
    h.ਬਿੱਟ = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    ਵਾਪਸ h;
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_FP32_ਮੰਜ਼ਿਲ_BF16(x) ggml_compute_fp32_ਮੰਜ਼ਿਲ_bf16(x)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_BF16_ਮੰਜ਼ਿਲ_FP32(x) ggml_compute_bf16_ਮੰਜ਼ਿਲ_fp32(x)

ਸਥਿਰ ਸਤਰਅੰਦਰ int32_t ggml_ਗੰਢ_get_use_ਗਿਣਤੀ(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * ਗੰਢ = cgraph->nodes[ਗੰਢ_idx];

    size_t hash_pos = ggml_hash_ਲੱਭੋ(&cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ, ਗੰਢ);
    ਜੇ (!ggml_bitset_get(cgraph->ਦੇਖਿਆ_hash_ਸੈੱਟ.ਵਰਤਿਆ, hash_pos)) {
        ਵਾਪਸ 0;
    }
    ਵਾਪਸ cgraph->use_counts[hash_pos];
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
ਸਥਿਰ ਸਤਰਅੰਦਰ bool ggml_ਗੰਢ_has_n_ਵਰਤਦਾ(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, int32_t n_ਵਰਤਦਾ) {
    ਅਚਲ ਬਣਤਰ ggml_tensor * ਗੰਢ = cgraph->nodes[ਗੰਢ_idx];

    // check the use count against how many we're replacing
    ਜੇ (ggml_ਗੰਢ_get_use_ਗਿਣਤੀ(cgraph, ਗੰਢ_idx) != n_ਵਰਤਦਾ) {
        ਵਾਪਸ false;
    }

    // if node is a view, some other node might be using the intermediate result
    // via the view source.
    ਜੇ (ਗੰਢ->view_src) {
        ਵਾਪਸ false;
    }

    // If the user requested output for the node, can't fuse
    ਜੇ (ਗੰਢ->flags & GGML_TENSOR_FLAG_OUTPUT) {
        ਵਾਪਸ false;
    }

    ਵਾਪਸ true;
}

// Returns true if nodes with indices { node_idxs } are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
ਸਥਿਰ ਸਤਰਅੰਦਰ bool ggml_can_fuse_ext(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਅਚਲ ਪੂਰਨ * ਗੰਢ_idxs, ਅਚਲ ਗਿਣਤੀ ggml_op * ops, ਪੂਰਨ ਅੰਕ_ops) {
    ਲਈ (ਪੂਰਨ i = 0; i < ਅੰਕ_ops; ++i) {
        ਜੇ (ਗੰਢ_idxs[i] >= cgraph->n_nodes) {
            ਵਾਪਸ false;
        }

        ਬਣਤਰ ggml_tensor * ਗੰਢ = cgraph->nodes[ਗੰਢ_idxs[i]];
        ਜੇ (ਗੰਢ->op != ops[i]) {
            ਵਾਪਸ false;
        }
        ਜੇ ((ਗੰਢ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ਵਾਪਸ false;
        }
        ਜੇ (i < ਅੰਕ_ops - 1 && !ggml_ਗੰਢ_has_n_ਵਰਤਦਾ(cgraph, ਗੰਢ_idxs[i], 1)) {
            ਵਾਪਸ false;
        }
        ਜੇ (i > 0) {
            ਬਣਤਰ ggml_tensor * prev = cgraph->nodes[ਗੰਢ_idxs[i - 1]];
            ਜੇ (ਗੰਢ->src[0] != prev && ਗੰਢ->src[1] != prev) {
                ਵਾਪਸ false;
            }
            ਜੇ (!ggml_are_same_shape(ਗੰਢ, prev)) {
                ਵਾਪਸ false;
            }
        }
    }
    ਵਾਪਸ true;
}

// same as above, for sequential indices starting at node_idx
ਸਥਿਰ ਸਤਰਅੰਦਰ bool ggml_can_fuse(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, ਅਚਲ ਗਿਣਤੀ ggml_op * ops, ਪੂਰਨ ਅੰਕ_ops) {
    ਪ੍ਰਤਿਜ੍ਞਾ(ਅੰਕ_ops < 32);

    ਜੇ (ਗੰਢ_idx + ਅੰਕ_ops > cgraph->n_nodes) {
        ਵਾਪਸ false;
    }

    ਪੂਰਨ idxs[32];
    ਲਈ (ਪੂਰਨ i = 0; i < ਅੰਕ_ops; ++i) {
        idxs[i] = ਗੰਢ_idx + i;
    }

    ਵਾਪਸ ggml_can_fuse_ext(cgraph, idxs, ops, ਅੰਕ_ops);
}

GGML_API bool ggml_can_fuse_subgraph_ext(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph,
                                         ਅਚਲ ਪੂਰਨ *                ਗੰਢ_idxs,
                                         ਪੂਰਨ                        ਗਿਣਤੀ_2,
                                         ਅਚਲ ਗਿਣਤੀ ggml_op *       ops,
                                         ਅਚਲ ਪੂਰਨ *                outputs,
                                         ਪੂਰਨ                        ਅੰਕ_outputs);

// Returns true if the subgraph formed by {node_idxs} can be fused
// checks whethers all nodes which are not part of outputs can be elided
// by checking if their num_uses are confined to the subgraph
ਸਥਿਰ ਸਤਰਅੰਦਰ bool ggml_can_fuse_subgraph(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph,
                                          ਪੂਰਨ                        ਗੰਢ_idx,
                                          ਪੂਰਨ                        ਗਿਣਤੀ_2,
                                          ਅਚਲ ਗਿਣਤੀ ggml_op *       ops,
                                          ਅਚਲ ਪੂਰਨ *                outputs,
                                          ਪੂਰਨ                        ਅੰਕ_outputs) {
    GGML_ASSERT(ਗਿਣਤੀ_2 < 32);
    ਜੇ (ਗੰਢ_idx + ਗਿਣਤੀ_2 > cgraph->n_nodes) {
        ਵਾਪਸ false;
    }

    ਪੂਰਨ idxs[32];

    ਲਈ (ਪੂਰਨ i = 0; i < ਗਿਣਤੀ_2; ++i) {
        idxs[i] = ਗੰਢ_idx + i;
    }

    ਵਾਪਸ ggml_can_fuse_subgraph_ext(cgraph, idxs, ਗਿਣਤੀ_2, ops, outputs, ਅੰਕ_outputs);
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
}
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
#ਅਨ੍ਤਰ੍ਭਾਵ <array>
#ਅਨ੍ਤਰ੍ਭਾਵ <initializer_list>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

// nicer C++ syntax for ggml_can_fuse
ਸਤਰਅੰਦਰ bool ggml_can_fuse(ਅਚਲ ਬਣਤਰ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, std::initializer_list<ਗਿਣਤੀ ggml_op> ops) {
    ਵਾਪਸ ggml_can_fuse(cgraph, ਗੰਢ_idx, ops.begin(), (ਪੂਰਨ)ops.ਮਾਪ());
}

ਸਤਰਅੰਦਰ bool ggml_can_fuse_subgraph(ਅਚਲ ਬਣਤਰ ggml_cgraph *          cgraph,
                                   ਪੂਰਨ                                 ਸ਼ੁਰੂ_idx,
                                   std::initializer_list<ਗਿਣਤੀ ggml_op> ops,
                                   std::initializer_list<ਪੂਰਨ>          outputs = {}) {
    ਵਾਪਸ ggml_can_fuse_subgraph(cgraph, ਸ਼ੁਰੂ_idx, ops.ਮਾਪ(), ops.begin(), outputs.begin(), outputs.ਮਾਪ());
}

// Return true if the edges in the graph match expectations.
ਸਤਰਅੰਦਰ bool ggml_check_ਕਿਨਾਰੇ(ਅਚਲ ਬਣਤਰ ggml_cgraph *                cgraph,
                             ਪੂਰਨ                                       ਸ਼ੁਰੂ_idx,
                             std::initializer_list<std::array<ਪੂਰਨ, 3>> ਕਿਨਾਰੇ) {
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ਕਿਨਾਰਾ : ਕਿਨਾਰੇ) {
        ਪੂਰਨ dst_ਗੰਢ = ਕਿਨਾਰਾ[0];
        ਪੂਰਨ src_idx  = ਕਿਨਾਰਾ[1];
        ਪੂਰਨ src_ਗੰਢ = ਕਿਨਾਰਾ[2];
        ਜੇ (cgraph->nodes[ਸ਼ੁਰੂ_idx + dst_ਗੰਢ]->src[src_idx] != cgraph->nodes[ਸ਼ੁਰੂ_idx + src_ਗੰਢ]) {
            ਵਾਪਸ false;
        }
    }
    ਵਾਪਸ true;
}

// expose GGUF internals for test code
GGML_API size_t gguf_type_ਮਾਪ(ਗਿਣਤੀ gguf_type type);
GGML_API ਖਾਲੀ gguf_write_ਮੰਜ਼ਿਲ_buf(ਅਚਲ ਬਣਤਰ gguf_context * ctx, std::ਸਦਿਸ਼<int8_t> & buf, bool only_meta);
#ਜੇ_ਅੰਤ // __cplusplus
