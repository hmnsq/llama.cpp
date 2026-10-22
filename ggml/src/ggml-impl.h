#ಸೂಚನೆ once

// GGML internal header

#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "gguf.h"

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __ARM_FEATURE_SVE
#ಅನ್ತರ್ಭಾವ <arm_sve.h>
#ಆದರೆ_ಅಂತ್ಯ // __ARM_FEATURE_SVE

#ಆದರೆ defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#ಅನ್ತರ್ಭಾವ <arm_neon.h>
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
ಬಾಹ್ಯ "C" {
#ಆದರೆ_ಅಂತ್ಯ

ಶೂನ್ಯ ggml_print_backtrace(ಶೂನ್ಯ);

uint64_t ggml_ನಕ್ಷೆ_ಮುಂದಿನ_uid(ಶೂನ್ಯ);

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ MIN
#    define MIN(a, b) ((a) < (b) ? (a) : (b))
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ MAX
#    define MAX(a, b) ((a) > (b) ? (a) : (b))
#ಆದರೆ_ಅಂತ್ಯ

// required for mmap as gguf only guarantees 32-byte alignment
#ವ್ಯಾಖ್ಯಾನಿಸು TENSOR_ALIGNMENT 32

// static_assert should be a #define, but if it's not,
// fall back to the _Static_assert C11 keyword.
// if C99 - static_assert is noop
// ref: https://stackoverflow.com/a/53923785/4039976
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ __cplusplus
    #ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ static_assert
        #ಆದರೆ defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
            #ವ್ಯಾಖ್ಯಾನಿಸು static_assert(cond, ಸಂದೇಶ_2) _Static_assert(cond, ಸಂದೇಶ_2)
        #ಇಲ್ಲದಿದ್ದರೆ
            #ವ್ಯಾಖ್ಯಾನಿಸು static_assert(cond, ಸಂದೇಶ_2) ರಚನೆ global_scope_noop_trick
        #ಆದರೆ_ಅಂತ್ಯ
    #ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ ggml_up32(ಪೂರ್ಣಾಂಕ n) {
    ಹಿಂತಿರುಗಿ (n + 31) & ~31;
}

//static inline int ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ ggml_ಮೇಲೆ(ಪೂರ್ಣಾಂಕ n, ಪೂರ್ಣಾಂಕ m) {
    // assert m is a power of 2
    GGML_ASSERT((m & (m - 1)) == 0);
    ಹಿಂತಿರುಗಿ (n + m - 1) & ~(m - 1);
}

// TODO: move to ggml.h? (won't be able to inline)
ಸ್ಥಿರ bool ggml_are_same_layout(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * a, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * b) {
    ಆದರೆ (a->type != b->type) {
        ಹಿಂತಿರುಗಿ false;
    }
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; i++) {
        ಆದರೆ (a->ne[i] != b->ne[i]) {
            ಹಿಂತಿರುಗಿ false;
        }
        ಆದರೆ (a->nb[i] != b->nb[i]) {
            ಹಿಂತಿರುಗಿ false;
        }
    }
    ಹಿಂತಿರುಗಿ true;
}

ಸ್ಥಿರ bool ggml_op_is_ಖಾಲಿಗಳು(ಎಣಿಕೆ ggml_op op) {
    ಆಯ್ಕೆ (op) {
        ಸ್ಥಿತಿ GGML_OP_NONE:
        ಸ್ಥಿತಿ GGML_OP_RESHAPE:
        ಸ್ಥಿತಿ GGML_OP_TRANSPOSE:
        ಸ್ಥಿತಿ GGML_OP_VIEW:
        ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ:
            ಹಿಂತಿರುಗಿ true;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ false;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool ggml_impl_is_view(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
    ಹಿಂತಿರುಗಿ t->view_src != ಖಾಲಿ;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ ggml_compute_softplus_f32(ದಶಮಾಂಶ ಒಳಹರಿವು) {
    ಹಿಂತಿರುಗಿ (ಒಳಹರಿವು > 20.0f) ? ಒಳಹರಿವು : logf(1 + expf(ಒಳಹರಿವು));
}
//
// logging
//

GGML_ATTRIBUTE_FORMAT(2, 3)
GGML_API ಶೂನ್ಯ ggml_log_ಆಂತರಿಕ        (ಎಣಿಕೆ ggml_log_level level, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * format, ...);
GGML_API ಶೂನ್ಯ ggml_log_callback_default(ಎಣಿಕೆ ggml_log_level level, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಪಠ್ಯ_2, ಶೂನ್ಯ * user_ದತ್ತಾಂಶ);

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_LOG(...)       ggml_log_ಆಂತರಿಕ(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_LOG_INFO(...)  ggml_log_ಆಂತರಿಕ(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_LOG_WARN(...)  ggml_log_ಆಂತರಿಕ(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_LOG_ERROR(...) ggml_log_ಆಂತರಿಕ(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_LOG_DEBUG(...) ggml_log_ಆಂತರಿಕ(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_LOG_CONT(...)  ggml_log_ಆಂತರಿಕ(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_DEBUG 0

#ಆದರೆ (GGML_DEBUG >= 1)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_PRINT_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_PRINT_DEBUG(...)
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ (GGML_DEBUG >= 5)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_PRINT_DEBUG_5(...) GGML_LOG_DEBUG(__VA_ARGS__)
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_PRINT_DEBUG_5(...)
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ (GGML_DEBUG >= 10)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_PRINT_DEBUG_10(...) GGML_LOG_DEBUG(__VA_ARGS__)
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_PRINT_DEBUG_10(...)
#ಆದರೆ_ಅಂತ್ಯ

// tensor params

ಸ್ಥಿರ ಶೂನ್ಯ ggml_ಗಣ_op_params(ರಚನೆ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * params, size_t params_ಗಾತ್ರ) {
    GGML_ASSERT(tensor != ಖಾಲಿ); // silence -Warray-bounds warnings
    ಪ್ರತಿಜ್ಞಾ(params_ಗಾತ್ರ <= GGML_MAX_OP_PARAMS);
    ಸ್ಮರಣೆ_ನಕಲು(tensor->op_params, params, params_ಗಾತ್ರ);
}

ಸ್ಥಿರ int32_t ggml_get_op_params_i32(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, uint32_t i) {
    ಪ್ರತಿಜ್ಞಾ(i < GGML_MAX_OP_PARAMS / ಗಾತ್ರ(int32_t));
    ಹಿಂತಿರುಗಿ ((ಸ್ಥಿರಾಂಕ int32_t *)(tensor->op_params))[i];
}

ಸ್ಥಿರ ದಶಮಾಂಶ ggml_get_op_params_f32(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, uint32_t i) {
    ಪ್ರತಿಜ್ಞಾ(i < GGML_MAX_OP_PARAMS / ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಹಿಂತಿರುಗಿ ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)(tensor->op_params))[i];
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
ಸ್ಥಿರ ಶೂನ್ಯ ggml_ಗಣ_op_params_i32(ರಚನೆ ggml_tensor * tensor, uint32_t i, int32_t ಮೌಲ್ಯ) {
    ಪ್ರತಿಜ್ಞಾ(i < GGML_MAX_OP_PARAMS / ಗಾತ್ರ(int32_t));
    ((int32_t *)(tensor->op_params))[i] = ಮೌಲ್ಯ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_ಗಣ_op_params_f32(ರಚನೆ ggml_tensor * tensor, uint32_t i, ದಶಮಾಂಶ ಮೌಲ್ಯ) {
    ಪ್ರತಿಜ್ಞಾ(i < GGML_MAX_OP_PARAMS / ಗಾತ್ರ(ದಶಮಾಂಶ));
    ((ದಶಮಾಂಶ *)(tensor->op_params))[i] = ಮೌಲ್ಯ;
}

ರಚನೆ ggml_map_custom1_op_params {
    ggml_custom1_op_t  fun;
    ಪೂರ್ಣಾಂಕ                n_tasks;
    ಶೂನ್ಯ             * userdata;
};

ರಚನೆ ggml_map_custom2_op_params {
    ggml_custom2_op_t   fun;
    ಪೂರ್ಣಾಂಕ                 n_tasks;
    ಶೂನ್ಯ              * userdata;
};

ರಚನೆ ggml_map_custom3_op_params {
    ggml_custom3_op_t fun;
    ಪೂರ್ಣಾಂಕ               n_tasks;
    ಶೂನ್ಯ            * userdata;
};

ರಚನೆ ggml_custom_op_params {
    ggml_custom_op_t fun;
    ಪೂರ್ಣಾಂಕ              n_tasks;
    ಶೂನ್ಯ           * userdata;
};

// bitset

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ uint32_t ggml_bitset_t;

static_assert(ಗಾತ್ರ(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#ವ್ಯಾಖ್ಯಾನಿಸು BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#ವ್ಯಾಖ್ಯಾನಿಸು BITSET_ಮುಖವಾಡ (ಗಾತ್ರ(ggml_bitset_t)*8 - 1)

ಸ್ಥಿರ size_t ggml_bitset_ಗಾತ್ರ(size_t n) {
    ಹಿಂತಿರುಗಿ (n + BITSET_ಮುಖವಾಡ) >> BITSET_SHR;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool ggml_bitset_get(ಸ್ಥಿರಾಂಕ ggml_bitset_t * bitset, size_t i) {
    ಹಿಂತಿರುಗಿ !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_ಮುಖವಾಡ)));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_bitset_ಗಣ(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_ಮುಖವಾಡ));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_bitset_clear(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_ಮುಖವಾಡ));
}

// hash set

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_HASHSET_ತುಂಬಿದ ((size_t)-1)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)

ರಚನೆ ggml_hash_ಗಣ {
    size_t ಗಾತ್ರ_2;
    ggml_bitset_t * ಬಳಸಿದ;       // whether or not the keys are in use i.e. set
    ರಚನೆ ggml_tensor ** ಕೀಲಿಗಳು; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};

ರಚನೆ ggml_hash_ಗಣ ggml_hash_ಗಣ_new(size_t ಗಾತ್ರ_2);
ಶೂನ್ಯ                 ggml_hash_ಗಣ_free(ರಚನೆ ggml_hash_ಗಣ * hash_ಗಣ);

// returns the minimum size for a hash set that can hold min_sz elements
size_t ggml_hash_ಗಾತ್ರ(size_t min_sz);

// remove all elements from the hash set
ಶೂನ್ಯ ggml_hash_ಗಣ_reset(ರಚನೆ ggml_hash_ಗಣ * hash_ಗಣ);

// returns true if key is in the hash set
ಸ್ಥಿರ bool ggml_hash_ಒಳಗೊಂಡಿದೆ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hash_ಗಣ * hash_ಗಣ, ರಚನೆ ggml_tensor * ಕೀಲಿ);

// returns GGML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
ಸ್ಥಿರ size_t ggml_hash_ಕಂಡುಹಿಡಿ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hash_ಗಣ * hash_ಗಣ, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಕೀಲಿ);

// returns GGML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
ಸ್ಥಿರ size_t ggml_hash_ಸೇರಿಸು(ರಚನೆ ggml_hash_ಗಣ * hash_ಗಣ, ರಚನೆ ggml_tensor * ಕೀಲಿ);

// return index, asserts if table is full
ಸ್ಥಿರ size_t ggml_hash_ಕಂಡುಹಿಡಿ_or_ಸೇರಿಸು(ರಚನೆ ggml_hash_ಗಣ * hash_ಗಣ, ರಚನೆ ggml_tensor * ಕೀಲಿ);

// hash function for ggml_tensor
ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t ggml_hash(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    ಹಿಂತಿರುಗಿ (size_t)(uintptr_t)p >> 4;
}

ಸ್ಥಿರ size_t ggml_hash_ಕಂಡುಹಿಡಿ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hash_ಗಣ * hash_ಗಣ, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಕೀಲಿ) {
    size_t h = ggml_hash(ಕೀಲಿ) % hash_ಗಣ->ಗಾತ್ರ_2;

    // linear probing
    size_t i = h;
    ಆಗ (ggml_bitset_get(hash_ಗಣ->ಬಳಸಿದ, i) && hash_ಗಣ->ಕೀಲಿಗಳು[i] != ಕೀಲಿ) {
        i = (i + 1) % hash_ಗಣ->ಗಾತ್ರ_2;
        ಆದರೆ (i == h) {
            // visited all hash table entries -> not found
            ಹಿಂತಿರುಗಿ GGML_HASHSET_ತುಂಬಿದ;
        }
    }
    ಹಿಂತಿರುಗಿ i;
}

ಸ್ಥಿರ bool ggml_hash_ಒಳಗೊಂಡಿದೆ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_hash_ಗಣ * hash_ಗಣ, ರಚನೆ ggml_tensor * ಕೀಲಿ) {
    size_t i = ggml_hash_ಕಂಡುಹಿಡಿ(hash_ಗಣ, ಕೀಲಿ);
    ಹಿಂತಿರುಗಿ i != GGML_HASHSET_ತುಂಬಿದ && ggml_bitset_get(hash_ಗಣ->ಬಳಸಿದ, i);
}

ಸ್ಥಿರ size_t ggml_hash_ಸೇರಿಸು(ರಚನೆ ggml_hash_ಗಣ * hash_ಗಣ, ರಚನೆ ggml_tensor * ಕೀಲಿ) {
    size_t h = ggml_hash(ಕೀಲಿ) % hash_ಗಣ->ಗಾತ್ರ_2;

    // linear probing
    size_t i = h;
    ಮಾಡು {
        ಆದರೆ (!ggml_bitset_get(hash_ಗಣ->ಬಳಸಿದ, i)) {
            ggml_bitset_ಗಣ(hash_ಗಣ->ಬಳಸಿದ, i);
            hash_ಗಣ->ಕೀಲಿಗಳು[i] = ಕೀಲಿ;
            ಹಿಂತಿರುಗಿ i;
        }
        ಆದರೆ (hash_ಗಣ->ಕೀಲಿಗಳು[i] == ಕೀಲಿ) {
            ಹಿಂತಿರುಗಿ GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_ಗಣ->ಗಾತ್ರ_2;
    } ಆಗ (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

ಸ್ಥಿರ size_t ggml_hash_ಕಂಡುಹಿಡಿ_or_ಸೇರಿಸು(ರಚನೆ ggml_hash_ಗಣ * hash_ಗಣ, ರಚನೆ ggml_tensor * ಕೀಲಿ) {
    size_t h = ggml_hash(ಕೀಲಿ) % hash_ಗಣ->ಗಾತ್ರ_2;

    // linear probing
    size_t i = h;
    ಮಾಡು {
        ಆದರೆ (!ggml_bitset_get(hash_ಗಣ->ಬಳಸಿದ, i)) {
            ggml_bitset_ಗಣ(hash_ಗಣ->ಬಳಸಿದ, i);
            hash_ಗಣ->ಕೀಲಿಗಳು[i] = ಕೀಲಿ;
            ಹಿಂತಿರುಗಿ i;
        }
        ಆದರೆ (hash_ಗಣ->ಕೀಲಿಗಳು[i] == ಕೀಲಿ) {
            ಹಿಂತಿರುಗಿ i;
        }
        i = (i + 1) % hash_ಗಣ->ಗಾತ್ರ_2;
    } ಆಗ (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

// computation graph

ಎಣಿಕೆ ggml_cgraph_eval_ಕ್ರಮ {
    GGML_CGRAPH_EVAL_ಕ್ರಮ_ಎಡ_ಗಮ್ಯ_ಬಲ = 0,
    GGML_CGRAPH_EVAL_ಕ್ರಮ_ಬಲ_ಗಮ್ಯ_ಎಡ,
    GGML_CGRAPH_EVAL_ಕ್ರಮ_ಎಣಿಕೆ
};

ರಚನೆ ggml_cgraph {
    ಪೂರ್ಣಾಂಕ ಗಾತ್ರ_2;    // maximum number of nodes/leafs/grads/grad_accs
    ಪೂರ್ಣಾಂಕ n_nodes; // number of nodes currently in use
    ಪೂರ್ಣಾಂಕ n_leafs; // number of leafs currently in use

    ರಚನೆ ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
    ರಚನೆ ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
    ರಚನೆ ggml_tensor ** grad_accs; // accumulators for node gradients
    ರಚನೆ ggml_tensor ** leafs;     // tensors with constant data
    int32_t             * use_counts;// number of uses of each tensor, indexed by hash table slot

    ರಚನೆ ggml_hash_ಗಣ ಭೇಟಿಯಾದ_hash_ಗಣ;

    ಎಣಿಕೆ ggml_cgraph_eval_ಕ್ರಮ ಕ್ರಮ;

    // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
    // a value of 0 means it is not set and should be ignored
    uint64_t uid;
};

// returns a slice of cgraph with nodes [i0, i1)
// the slice does not have leafs or gradients
// if you need the gradients, get them from the original graph
ರಚನೆ ggml_cgraph ggml_ನಕ್ಷೆ_view(ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ i0, ಪೂರ್ಣಾಂಕ i1);

// ggml-alloc.c: true if the operation can reuse memory from its sources
GGML_API bool ggml_op_can_inplace(ಎಣಿಕೆ ggml_op op);


// Memory allocation

GGML_API ಶೂನ್ಯ * ggml_aligned_malloc(size_t ಗಾತ್ರ_2);
GGML_API ಶೂನ್ಯ ggml_aligned_free(ಶೂನ್ಯ * ptr, size_t ಗಾತ್ರ_2);

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ fp32_from_ಬಿಟ್ಟುಗಳು(uint32_t w) {
    ಒಕ್ಕೂಟ {
        uint32_t as_ಬಿಟ್ಟುಗಳು;
        ದಶಮಾಂಶ as_ಮೌಲ್ಯ;
    } fp32;
    fp32.as_ಬಿಟ್ಟುಗಳು = w;
    ಹಿಂತಿರುಗಿ fp32.as_ಮೌಲ್ಯ;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t fp32_ಗಮ್ಯ_ಬಿಟ್ಟುಗಳು(ದಶಮಾಂಶ f) {
    ಒಕ್ಕೂಟ {
        ದಶಮಾಂಶ as_ಮೌಲ್ಯ;
        uint32_t as_ಬಿಟ್ಟುಗಳು;
    } fp32;
    fp32.as_ಮೌಲ್ಯ = f;
    ಹಿಂತಿರುಗಿ fp32.as_ಬಿಟ್ಟುಗಳು;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ ggml_compute_fp16_ಗಮ್ಯ_fp32(ggml_fp16_t h) {
    ಸ್ಥಿರಾಂಕ uint32_t w = (uint32_t) h << 16;
    ಸ್ಥಿರಾಂಕ uint32_t sign = w & UINT32_C(0x80000000);
    ಸ್ಥಿರಾಂಕ uint32_t two_w = w + w;

    ಸ್ಥಿರಾಂಕ uint32_t exp_ಸ್ಥಾನಾಂತರ = UINT32_C(0xE0) << 23;
#ಆದರೆ (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ exp_scale = 0x1.0p-112f;
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ exp_scale = fp32_from_ಬಿಟ್ಟುಗಳು(UINT32_C(0x7800000));
#ಆದರೆ_ಅಂತ್ಯ
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ normalized_ಮೌಲ್ಯ = fp32_from_ಬಿಟ್ಟುಗಳು((two_w >> 4) + exp_ಸ್ಥಾನಾಂತರ) * exp_scale;

    ಸ್ಥಿರಾಂಕ uint32_t magic_ಮುಖವಾಡ = UINT32_C(126) << 23;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ magic_bias = 0.5f;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ denormalized_ಮೌಲ್ಯ = fp32_from_ಬಿಟ್ಟುಗಳು((two_w >> 17) | magic_ಮುಖವಾಡ) - magic_bias;

    ಸ್ಥಿರಾಂಕ uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    ಸ್ಥಿರಾಂಕ uint32_t ಫಲಿತಾಂಶ = sign |
        (two_w < denormalized_cutoff ? fp32_ಗಮ್ಯ_ಬಿಟ್ಟುಗಳು(denormalized_ಮೌಲ್ಯ) : fp32_ಗಮ್ಯ_ಬಿಟ್ಟುಗಳು(normalized_ಮೌಲ್ಯ));
    ಹಿಂತಿರುಗಿ fp32_from_ಬಿಟ್ಟುಗಳು(ಫಲಿತಾಂಶ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ggml_fp16_t ggml_compute_fp32_ಗಮ್ಯ_fp16(ದಶಮಾಂಶ f) {
#ಆದರೆ (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale_ಗಮ್ಯ_inf = 0x1.0p+112f;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale_ಗಮ್ಯ_zero = 0x1.0p-110f;
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale_ಗಮ್ಯ_inf = fp32_from_ಬಿಟ್ಟುಗಳು(UINT32_C(0x77800000));
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale_ಗಮ್ಯ_zero = fp32_from_ಬಿಟ್ಟುಗಳು(UINT32_C(0x08800000));
#ಆದರೆ_ಅಂತ್ಯ
    ದಶಮಾಂಶ ತಳ = (fabsf(f) * scale_ಗಮ್ಯ_inf) * scale_ಗಮ್ಯ_zero;

    ಸ್ಥಿರಾಂಕ uint32_t w = fp32_ಗಮ್ಯ_ಬಿಟ್ಟುಗಳು(f);
    ಸ್ಥಿರಾಂಕ uint32_t shl1_w = w + w;
    ಸ್ಥಿರಾಂಕ uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    ಆದರೆ (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    ತಳ = fp32_from_ಬಿಟ್ಟುಗಳು((bias >> 1) + UINT32_C(0x07800000)) + ತಳ;
    ಸ್ಥಿರಾಂಕ uint32_t ಬಿಟ್ಟುಗಳು = fp32_ಗಮ್ಯ_ಬಿಟ್ಟುಗಳು(ತಳ);
    ಸ್ಥಿರಾಂಕ uint32_t exp_ಬಿಟ್ಟುಗಳು = (ಬಿಟ್ಟುಗಳು >> 13) & UINT32_C(0x00007C00);
    ಸ್ಥಿರಾಂಕ uint32_t mantissa_ಬಿಟ್ಟುಗಳು = ಬಿಟ್ಟುಗಳು & UINT32_C(0x00000FFF);
    ಸ್ಥಿರಾಂಕ uint32_t nonsign = exp_ಬಿಟ್ಟುಗಳು + mantissa_ಬಿಟ್ಟುಗಳು;
    ಹಿಂತಿರುಗಿ (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMPUTE_FP16_ಗಮ್ಯ_FP32(x) ggml_compute_fp16_ಗಮ್ಯ_fp32(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMPUTE_FP32_ಗಮ್ಯ_FP16(x) ggml_compute_fp32_ಗಮ್ಯ_fp16(x)

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_FP16_ಗಮ್ಯ_FP32(x) GGML_COMPUTE_FP16_ಗಮ್ಯ_FP32(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_FP32_ಗಮ್ಯ_FP16(x) GGML_COMPUTE_FP32_ಗಮ್ಯ_FP16(x)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ ggml_e8m0_ಗಮ್ಯ_fp32(uint8_t x) {
    uint32_t ಬಿಟ್ಟುಗಳು;  // Stores the raw bit representation of the float

    // Handle special case for minimum exponent (denormalized float)
    ಆದರೆ (x == 0) {
        // Bit pattern for 2^(-127):
        // - Sign bit: 0 (positive)
        // - Exponent: 0 (denormalized number)
        // - Mantissa: 0x400000 (0.5 in fractional form)
        // Value = 0.5 * 2^(-126) = 2^(-127)
        ಬಿಟ್ಟುಗಳು = 0x00400000;
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
    ಇಲ್ಲದಿದ್ದರೆ {
        // Construct normalized float by shifting exponent into position:
        // - Exponent field: 8 bits (positions 30-23)
        // - Mantissa: 0 (implicit leading 1)
        // Value = 2^(x - 127)
        ಬಿಟ್ಟುಗಳು = (uint32_t) x << 23;
    }

    ದಶಮಾಂಶ ಫಲಿತಾಂಶ;  // Final float value
                   // Safely reinterpret bit pattern as float without type-punning issues
    ಸ್ಮರಣೆ_ನಕಲು(&ಫಲಿತಾಂಶ, &ಬಿಟ್ಟುಗಳು, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ ggml_e8m0_ಗಮ್ಯ_fp32_half(uint8_t x) {
    uint32_t ಬಿಟ್ಟುಗಳು;

    // For x < 2: use precomputed denormal patterns
    ಆದರೆ (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        ಬಿಟ್ಟುಗಳು = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    ಇಲ್ಲದಿದ್ದರೆ {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        ಬಿಟ್ಟುಗಳು = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    ದಶಮಾಂಶ ಫಲಿತಾಂಶ;
    ಸ್ಮರಣೆ_ನಕಲು(&ಫಲಿತಾಂಶ, &ಬಿಟ್ಟುಗಳು, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_E8_M0_ಗಮ್ಯ_FP32(x) ggml_e8m0_ಗಮ್ಯ_fp32(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_E8_M0_ಗಮ್ಯ_FP32_HALF(x) ggml_e8m0_ಗಮ್ಯ_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ ggml_ue4m3_ಗಮ್ಯ_fp32(uint8_t x) {
    ಆದರೆ (x == 0 || x == 0x7F) {
        ಹಿಂತಿರುಗಿ 0.0f;
    }
    ಪೂರ್ಣಾಂಕ   ಘಾತಾಂಕ = (x >> 3) & 0xF;
    ಪೂರ್ಣಾಂಕ   man = x & 0x7;
    ದಶಮಾಂಶ raw;
    ಆದರೆ (ಘಾತಾಂಕ == 0) {
        raw = ldexpf((ದಶಮಾಂಶ) man, -9);
    } ಇಲ್ಲದಿದ್ದರೆ {
        raw = ldexpf(1.0f + (ದಶಮಾಂಶ) man / 8.0f, ಘಾತಾಂಕ - 7);
    }
    ಹಿಂತಿರುಗಿ raw * 0.5f;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint8_t ggml_fp32_ಗಮ್ಯ_ue4m3(ದಶಮಾಂಶ x) {
    ಆದರೆ (!(x > 0.0f)) {
        ಹಿಂತಿರುಗಿ 0;
    }
    ಆದರೆ (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t ಬಿಟ್ಟುಗಳು;
    ಸ್ಮರಣೆ_ನಕಲು(&ಬಿಟ್ಟುಗಳು, &x, 4);
    ಪೂರ್ಣಾಂಕ fp32_exp  = ((ಬಿಟ್ಟುಗಳು >> 23) & 0xFF) - 127;
    ಪೂರ್ಣಾಂಕ fp32_man  = (ಬಿಟ್ಟುಗಳು >> 20) & 0x7;
    ಪೂರ್ಣಾಂಕ ue4m3_exp = fp32_exp + 7;
    ಆದರೆ (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        ಪೂರ್ಣಾಂಕ man = (ಪೂರ್ಣಾಂಕ) (x * 512.0f + 0.5f);
        ಆದರೆ (man > 7) {
            man = 7;
        }
        ಆದರೆ (man < 1) {
            ಹಿಂತಿರುಗಿ 0;
        }
        ಹಿಂತಿರುಗಿ (uint8_t) man;
    }
    ಆದರೆ (ue4m3_exp >= 15) {
        ಹಿಂತಿರುಗಿ 0x7E;
    }
    ಪೂರ್ಣಾಂಕ ಸುತ್ತು_ಬಿಟ್ = (ಬಿಟ್ಟುಗಳು >> 19) & 1;
    ಪೂರ್ಣಾಂಕ ue4m3_man = fp32_man + ಸುತ್ತು_ಬಿಟ್;
    ಆದರೆ (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        ಆದರೆ (ue4m3_exp >= 15) {
            ಹಿಂತಿರುಗಿ 0x7E;
        }
    }
    ಹಿಂತಿರುಗಿ (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
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
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ ggml_compute_bf16_ಗಮ್ಯ_fp32(ggml_bf16_t h) {
    ಒಕ್ಕೂಟ {
        ದಶಮಾಂಶ f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.ಬಿಟ್ಟುಗಳು << 16;
    ಹಿಂತಿರುಗಿ u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ggml_bf16_t ggml_compute_fp32_ಗಮ್ಯ_bf16(ದಶಮಾಂಶ s) {
    ggml_bf16_t h;
    ಒಕ್ಕೂಟ {
        ದಶಮಾಂಶ f;
        uint32_t i;
    } u;
    u.f = s;
    ಆದರೆ ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.ಬಿಟ್ಟುಗಳು = (u.i >> 16) | 64; /* force to quiet */
        ಹಿಂತಿರುಗಿ h;
    }
    h.ಬಿಟ್ಟುಗಳು = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    ಹಿಂತಿರುಗಿ h;
}

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_FP32_ಗಮ್ಯ_BF16(x) ggml_compute_fp32_ಗಮ್ಯ_bf16(x)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_BF16_ಗಮ್ಯ_FP32(x) ggml_compute_bf16_ಗಮ್ಯ_fp32(x)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ int32_t ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಗಂಟು = cgraph->nodes[ಗಂಟು_idx];

    size_t hash_pos = ggml_hash_ಕಂಡುಹಿಡಿ(&cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ, ಗಂಟು);
    ಆದರೆ (!ggml_bitset_get(cgraph->ಭೇಟಿಯಾದ_hash_ಗಣ.ಬಳಸಿದ, hash_pos)) {
        ಹಿಂತಿರುಗಿ 0;
    }
    ಹಿಂತಿರುಗಿ cgraph->use_counts[hash_pos];
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool ggml_ಗಂಟು_has_n_ಬಳಸುತ್ತದೆ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, int32_t n_ಬಳಸುತ್ತದೆ) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಗಂಟು = cgraph->nodes[ಗಂಟು_idx];

    // check the use count against how many we're replacing
    ಆದರೆ (ggml_ಗಂಟು_get_use_ಎಣಿಕೆ(cgraph, ಗಂಟು_idx) != n_ಬಳಸುತ್ತದೆ) {
        ಹಿಂತಿರುಗಿ false;
    }

    // if node is a view, some other node might be using the intermediate result
    // via the view source.
    ಆದರೆ (ಗಂಟು->view_src) {
        ಹಿಂತಿರುಗಿ false;
    }

    // If the user requested output for the node, can't fuse
    ಆದರೆ (ಗಂಟು->flags & GGML_TENSOR_FLAG_OUTPUT) {
        ಹಿಂತಿರುಗಿ false;
    }

    ಹಿಂತಿರುಗಿ true;
}

// Returns true if nodes with indices { node_idxs } are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool ggml_can_fuse_ext(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * ಗಂಟು_idxs, ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_op * ops, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ops) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಅಂಕೆ_ops; ++i) {
        ಆದರೆ (ಗಂಟು_idxs[i] >= cgraph->n_nodes) {
            ಹಿಂತಿರುಗಿ false;
        }

        ರಚನೆ ggml_tensor * ಗಂಟು = cgraph->nodes[ಗಂಟು_idxs[i]];
        ಆದರೆ (ಗಂಟು->op != ops[i]) {
            ಹಿಂತಿರುಗಿ false;
        }
        ಆದರೆ ((ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ಹಿಂತಿರುಗಿ false;
        }
        ಆದರೆ (i < ಅಂಕೆ_ops - 1 && !ggml_ಗಂಟು_has_n_ಬಳಸುತ್ತದೆ(cgraph, ಗಂಟು_idxs[i], 1)) {
            ಹಿಂತಿರುಗಿ false;
        }
        ಆದರೆ (i > 0) {
            ರಚನೆ ggml_tensor * prev = cgraph->nodes[ಗಂಟು_idxs[i - 1]];
            ಆದರೆ (ಗಂಟು->src[0] != prev && ಗಂಟು->src[1] != prev) {
                ಹಿಂತಿರುಗಿ false;
            }
            ಆದರೆ (!ggml_are_same_shape(ಗಂಟು, prev)) {
                ಹಿಂತಿರುಗಿ false;
            }
        }
    }
    ಹಿಂತಿರುಗಿ true;
}

// same as above, for sequential indices starting at node_idx
ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool ggml_can_fuse(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_op * ops, ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ops) {
    ಪ್ರತಿಜ್ಞಾ(ಅಂಕೆ_ops < 32);

    ಆದರೆ (ಗಂಟು_idx + ಅಂಕೆ_ops > cgraph->n_nodes) {
        ಹಿಂತಿರುಗಿ false;
    }

    ಪೂರ್ಣಾಂಕ idxs[32];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಅಂಕೆ_ops; ++i) {
        idxs[i] = ಗಂಟು_idx + i;
    }

    ಹಿಂತಿರುಗಿ ggml_can_fuse_ext(cgraph, idxs, ops, ಅಂಕೆ_ops);
}

GGML_API bool ggml_can_fuse_subgraph_ext(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph,
                                         ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *                ಗಂಟು_idxs,
                                         ಪೂರ್ಣಾಂಕ                        ಎಣಿಕೆ_2,
                                         ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_op *       ops,
                                         ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *                outputs,
                                         ಪೂರ್ಣಾಂಕ                        ಅಂಕೆ_outputs);

// Returns true if the subgraph formed by {node_idxs} can be fused
// checks whethers all nodes which are not part of outputs can be elided
// by checking if their num_uses are confined to the subgraph
ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool ggml_can_fuse_subgraph(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph,
                                          ಪೂರ್ಣಾಂಕ                        ಗಂಟು_idx,
                                          ಪೂರ್ಣಾಂಕ                        ಎಣಿಕೆ_2,
                                          ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_op *       ops,
                                          ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *                outputs,
                                          ಪೂರ್ಣಾಂಕ                        ಅಂಕೆ_outputs) {
    GGML_ASSERT(ಎಣಿಕೆ_2 < 32);
    ಆದರೆ (ಗಂಟು_idx + ಎಣಿಕೆ_2 > cgraph->n_nodes) {
        ಹಿಂತಿರುಗಿ false;
    }

    ಪೂರ್ಣಾಂಕ idxs[32];

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಎಣಿಕೆ_2; ++i) {
        idxs[i] = ಗಂಟು_idx + i;
    }

    ಹಿಂತಿರುಗಿ ggml_can_fuse_subgraph_ext(cgraph, idxs, ಎಣಿಕೆ_2, ops, outputs, ಅಂಕೆ_outputs);
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
}
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
#ಅನ್ತರ್ಭಾವ <array>
#ಅನ್ತರ್ಭಾವ <initializer_list>
#ಅನ್ತರ್ಭಾವ <vector>

// nicer C++ syntax for ggml_can_fuse
ಸಾಲೊಳಗೆ bool ggml_can_fuse(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, std::initializer_list<ಎಣಿಕೆ ggml_op> ops) {
    ಹಿಂತಿರುಗಿ ggml_can_fuse(cgraph, ಗಂಟು_idx, ops.begin(), (ಪೂರ್ಣಾಂಕ)ops.ಗಾತ್ರ_2());
}

ಸಾಲೊಳಗೆ bool ggml_can_fuse_subgraph(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph *          cgraph,
                                   ಪೂರ್ಣಾಂಕ                                 ಆರಂಭ_idx,
                                   std::initializer_list<ಎಣಿಕೆ ggml_op> ops,
                                   std::initializer_list<ಪೂರ್ಣಾಂಕ>          outputs = {}) {
    ಹಿಂತಿರುಗಿ ggml_can_fuse_subgraph(cgraph, ಆರಂಭ_idx, ops.ಗಾತ್ರ_2(), ops.begin(), outputs.begin(), outputs.ಗಾತ್ರ_2());
}

// Return true if the edges in the graph match expectations.
ಸಾಲೊಳಗೆ bool ggml_check_ಅಂಚುಗಳು(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_cgraph *                cgraph,
                             ಪೂರ್ಣಾಂಕ                                       ಆರಂಭ_idx,
                             std::initializer_list<std::array<ಪೂರ್ಣಾಂಕ, 3>> ಅಂಚುಗಳು) {
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಅಂಚು : ಅಂಚುಗಳು) {
        ಪೂರ್ಣಾಂಕ dst_ಗಂಟು = ಅಂಚು[0];
        ಪೂರ್ಣಾಂಕ src_idx  = ಅಂಚು[1];
        ಪೂರ್ಣಾಂಕ src_ಗಂಟು = ಅಂಚು[2];
        ಆದರೆ (cgraph->nodes[ಆರಂಭ_idx + dst_ಗಂಟು]->src[src_idx] != cgraph->nodes[ಆರಂಭ_idx + src_ಗಂಟು]) {
            ಹಿಂತಿರುಗಿ false;
        }
    }
    ಹಿಂತಿರುಗಿ true;
}

// expose GGUF internals for test code
GGML_API size_t gguf_type_ಗಾತ್ರ(ಎಣಿಕೆ gguf_type type);
GGML_API ಶೂನ್ಯ gguf_write_ಗಮ್ಯ_buf(ಸ್ಥಿರಾಂಕ ರಚನೆ gguf_context * ctx, std::ಸದಿಶ<int8_t> & buf, bool only_meta);
#ಆದರೆ_ಅಂತ್ಯ // __cplusplus
