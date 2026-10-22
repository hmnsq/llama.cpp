#pragma once

// GGML internal header

#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "gguf.h"

#അന്തര്ഭാവ <assert.h>
#അന്തര്ഭാവ <math.h>
#അന്തര്ഭാവ <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#അന്തര്ഭാവ <stdbool.h>
#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <string.h>

#ifdef __ARM_FEATURE_SVE
#അന്തര്ഭാവ <arm_sve.h>
#endif // __ARM_FEATURE_SVE

#എങ്കിൽ defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#അന്തര്ഭാവ <arm_neon.h>
#endif

#ifdef __cplusplus
ബാഹ്യ "C" {
#endif

ശൂന്യം ggml_print_backtrace(ശൂന്യം);

uint64_t ggml_graph_next_uid(ശൂന്യം);

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
        #എങ്കിൽ defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
            #define static_assert(cond, സന്ദേശം) _Static_assert(cond, സന്ദേശം)
        #അല്ലെങ്കിൽ
            #define static_assert(cond, സന്ദേശം) ഘടന global_scope_noop_trick
        #endif
    #endif
#endif

സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ ggml_up32(പൂർണ്ണസംഖ്യ n) {
    തിരികെ (n + 31) & ~31;
}

//static inline int ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ ggml_up(പൂർണ്ണസംഖ്യ n, പൂർണ്ണസംഖ്യ m) {
    // assert m is a power of 2
    GGML_ASSERT((m & (m - 1)) == 0);
    തിരികെ (n + m - 1) & ~(m - 1);
}

// TODO: move to ggml.h? (won't be able to inline)
സ്ഥിതം bool ggml_are_same_layout(സ്ഥിരം ഘടന ggml_tensor * a, സ്ഥിരം ഘടന ggml_tensor * b) {
    എങ്കിൽ (a->type != b->type) {
        തിരികെ false;
    }
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < GGML_MAX_DIMS; i++) {
        എങ്കിൽ (a->ne[i] != b->ne[i]) {
            തിരികെ false;
        }
        എങ്കിൽ (a->nb[i] != b->nb[i]) {
            തിരികെ false;
        }
    }
    തിരികെ true;
}

സ്ഥിതം bool ggml_op_is_empty(എണ്ണൽ ggml_op op) {
    തിരഞ്ഞെടുപ്പ് (op) {
        അവസ്ഥ GGML_OP_NONE:
        അവസ്ഥ GGML_OP_RESHAPE:
        അവസ്ഥ GGML_OP_TRANSPOSE:
        അവസ്ഥ GGML_OP_VIEW:
        അവസ്ഥ GGML_OP_PERMUTE:
            തിരികെ true;
        സ്ഥിരസ്ഥിതി:
            തിരികെ false;
    }
}

സ്ഥിതം വരിയിൽ bool ggml_impl_is_view(സ്ഥിരം ഘടന ggml_tensor * t) {
    തിരികെ t->view_src != ശൂന്യ;
}

സ്ഥിതം വരിയിൽ ദശാംശം ggml_compute_softplus_f32(ദശാംശം നിവേശം) {
    തിരികെ (നിവേശം > 20.0f) ? നിവേശം : logf(1 + expf(നിവേശം));
}
//
// logging
//

GGML_ATTRIBUTE_FORMAT(2, 3)
GGML_API ശൂന്യം ggml_log_internal        (എണ്ണൽ ggml_log_level level, സ്ഥിരം അക്ഷരം * format, ...);
GGML_API ശൂന്യം ggml_log_callback_default(എണ്ണൽ ggml_log_level level, സ്ഥിരം അക്ഷരം * text, ശൂന്യം * user_data);

#define GGML_LOG(...)       ggml_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define GGML_LOG_INFO(...)  ggml_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define GGML_LOG_WARN(...)  ggml_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define GGML_LOG_ERROR(...) ggml_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GGML_LOG_DEBUG(...) ggml_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define GGML_LOG_CONT(...)  ggml_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

#define GGML_DEBUG 0

#എങ്കിൽ (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#അല്ലെങ്കിൽ
#define GGML_PRINT_DEBUG(...)
#endif

#എങ്കിൽ (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) GGML_LOG_DEBUG(__VA_ARGS__)
#അല്ലെങ്കിൽ
#define GGML_PRINT_DEBUG_5(...)
#endif

#എങ്കിൽ (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) GGML_LOG_DEBUG(__VA_ARGS__)
#അല്ലെങ്കിൽ
#define GGML_PRINT_DEBUG_10(...)
#endif

// tensor params

സ്ഥിതം ശൂന്യം ggml_set_op_params(ഘടന ggml_tensor * tensor, സ്ഥിരം ശൂന്യം * params, size_t params_size) {
    GGML_ASSERT(tensor != ശൂന്യ); // silence -Warray-bounds warnings
    പ്രതിജ്ഞാ(params_size <= GGML_MAX_OP_PARAMS);
    memcpy(tensor->op_params, params, params_size);
}

സ്ഥിതം int32_t ggml_get_op_params_i32(സ്ഥിരം ഘടന ggml_tensor * tensor, uint32_t i) {
    പ്രതിജ്ഞാ(i < GGML_MAX_OP_PARAMS / വലുപ്പം(int32_t));
    തിരികെ ((സ്ഥിരം int32_t *)(tensor->op_params))[i];
}

സ്ഥിതം ദശാംശം ggml_get_op_params_f32(സ്ഥിരം ഘടന ggml_tensor * tensor, uint32_t i) {
    പ്രതിജ്ഞാ(i < GGML_MAX_OP_PARAMS / വലുപ്പം(ദശാംശം));
    തിരികെ ((സ്ഥിരം ദശാംശം *)(tensor->op_params))[i];
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
സ്ഥിതം ശൂന്യം ggml_set_op_params_i32(ഘടന ggml_tensor * tensor, uint32_t i, int32_t മൂല്യം) {
    പ്രതിജ്ഞാ(i < GGML_MAX_OP_PARAMS / വലുപ്പം(int32_t));
    ((int32_t *)(tensor->op_params))[i] = മൂല്യം;
}

സ്ഥിതം ശൂന്യം ggml_set_op_params_f32(ഘടന ggml_tensor * tensor, uint32_t i, ദശാംശം മൂല്യം) {
    പ്രതിജ്ഞാ(i < GGML_MAX_OP_PARAMS / വലുപ്പം(ദശാംശം));
    ((ദശാംശം *)(tensor->op_params))[i] = മൂല്യം;
}

ഘടന ggml_map_custom1_op_params {
    ggml_custom1_op_t  fun;
    പൂർണ്ണസംഖ്യ                n_tasks;
    ശൂന്യം             * userdata;
};

ഘടന ggml_map_custom2_op_params {
    ggml_custom2_op_t   fun;
    പൂർണ്ണസംഖ്യ                 n_tasks;
    ശൂന്യം              * userdata;
};

ഘടന ggml_map_custom3_op_params {
    ggml_custom3_op_t fun;
    പൂർണ്ണസംഖ്യ               n_tasks;
    ശൂന്യം            * userdata;
};

ഘടന ggml_custom_op_params {
    ggml_custom_op_t fun;
    പൂർണ്ണസംഖ്യ              n_tasks;
    ശൂന്യം           * userdata;
};

// bitset

തരനിർവചനം uint32_t ggml_bitset_t;

static_assert(വലുപ്പം(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#define BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#define BITSET_MASK (വലുപ്പം(ggml_bitset_t)*8 - 1)

സ്ഥിതം size_t ggml_bitset_size(size_t n) {
    തിരികെ (n + BITSET_MASK) >> BITSET_SHR;
}

സ്ഥിതം വരിയിൽ bool ggml_bitset_get(സ്ഥിരം ggml_bitset_t * bitset, size_t i) {
    തിരികെ !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_bitset_set(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_MASK));
}

സ്ഥിതം വരിയിൽ ശൂന്യം ggml_bitset_clear(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_MASK));
}

// hash set

#define GGML_HASHSET_FULL ((size_t)-1)
#define GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)

ഘടന ggml_hash_set {
    size_t size;
    ggml_bitset_t * used;       // whether or not the keys are in use i.e. set
    ഘടന ggml_tensor ** താക്കോലുകൾ; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};

ഘടന ggml_hash_set ggml_hash_set_new(size_t size);
ശൂന്യം                 ggml_hash_set_free(ഘടന ggml_hash_set * hash_set);

// returns the minimum size for a hash set that can hold min_sz elements
size_t ggml_hash_size(size_t min_sz);

// remove all elements from the hash set
ശൂന്യം ggml_hash_set_reset(ഘടന ggml_hash_set * hash_set);

// returns true if key is in the hash set
സ്ഥിതം bool ggml_hash_contains(സ്ഥിരം ഘടന ggml_hash_set * hash_set, ഘടന ggml_tensor * താക്കോൽ);

// returns GGML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
സ്ഥിതം size_t ggml_hash_find(സ്ഥിരം ഘടന ggml_hash_set * hash_set, സ്ഥിരം ഘടന ggml_tensor * താക്കോൽ);

// returns GGML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
സ്ഥിതം size_t ggml_hash_തിരുകുക(ഘടന ggml_hash_set * hash_set, ഘടന ggml_tensor * താക്കോൽ);

// return index, asserts if table is full
സ്ഥിതം size_t ggml_hash_find_or_തിരുകുക(ഘടന ggml_hash_set * hash_set, ഘടന ggml_tensor * താക്കോൽ);

// hash function for ggml_tensor
സ്ഥിതം വരിയിൽ size_t ggml_hash(സ്ഥിരം ഘടന ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    തിരികെ (size_t)(uintptr_t)p >> 4;
}

സ്ഥിതം size_t ggml_hash_find(സ്ഥിരം ഘടന ggml_hash_set * hash_set, സ്ഥിരം ഘടന ggml_tensor * താക്കോൽ) {
    size_t h = ggml_hash(താക്കോൽ) % hash_set->size;

    // linear probing
    size_t i = h;
    ആകുമ്പോൾ (ggml_bitset_get(hash_set->used, i) && hash_set->താക്കോലുകൾ[i] != താക്കോൽ) {
        i = (i + 1) % hash_set->size;
        എങ്കിൽ (i == h) {
            // visited all hash table entries -> not found
            തിരികെ GGML_HASHSET_FULL;
        }
    }
    തിരികെ i;
}

സ്ഥിതം bool ggml_hash_contains(സ്ഥിരം ഘടന ggml_hash_set * hash_set, ഘടന ggml_tensor * താക്കോൽ) {
    size_t i = ggml_hash_find(hash_set, താക്കോൽ);
    തിരികെ i != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, i);
}

സ്ഥിതം size_t ggml_hash_തിരുകുക(ഘടന ggml_hash_set * hash_set, ഘടന ggml_tensor * താക്കോൽ) {
    size_t h = ggml_hash(താക്കോൽ) % hash_set->size;

    // linear probing
    size_t i = h;
    ചെയ്യുക {
        എങ്കിൽ (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->താക്കോലുകൾ[i] = താക്കോൽ;
            തിരികെ i;
        }
        എങ്കിൽ (hash_set->താക്കോലുകൾ[i] == താക്കോൽ) {
            തിരികെ GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_set->size;
    } ആകുമ്പോൾ (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

സ്ഥിതം size_t ggml_hash_find_or_തിരുകുക(ഘടന ggml_hash_set * hash_set, ഘടന ggml_tensor * താക്കോൽ) {
    size_t h = ggml_hash(താക്കോൽ) % hash_set->size;

    // linear probing
    size_t i = h;
    ചെയ്യുക {
        എങ്കിൽ (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->താക്കോലുകൾ[i] = താക്കോൽ;
            തിരികെ i;
        }
        എങ്കിൽ (hash_set->താക്കോലുകൾ[i] == താക്കോൽ) {
            തിരികെ i;
        }
        i = (i + 1) % hash_set->size;
    } ആകുമ്പോൾ (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

// computation graph

എണ്ണൽ ggml_cgraph_eval_order {
    GGML_CGRAPH_EVAL_ORDER_ഇടത്_TO_വലത് = 0,
    GGML_CGRAPH_EVAL_ORDER_വലത്_TO_ഇടത്,
    GGML_CGRAPH_EVAL_ORDER_എണ്ണം
};

ഘടന ggml_cgraph {
    പൂർണ്ണസംഖ്യ size;    // maximum number of nodes/leafs/grads/grad_accs
    പൂർണ്ണസംഖ്യ n_nodes; // number of nodes currently in use
    പൂർണ്ണസംഖ്യ n_leafs; // number of leafs currently in use

    ഘടന ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
    ഘടന ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
    ഘടന ggml_tensor ** grad_accs; // accumulators for node gradients
    ഘടന ggml_tensor ** leafs;     // tensors with constant data
    int32_t             * use_counts;// number of uses of each tensor, indexed by hash table slot

    ഘടന ggml_hash_set visited_hash_set;

    എണ്ണൽ ggml_cgraph_eval_order order;

    // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
    // a value of 0 means it is not set and should be ignored
    uint64_t uid;
};

// returns a slice of cgraph with nodes [i0, i1)
// the slice does not have leafs or gradients
// if you need the gradients, get them from the original graph
ഘടന ggml_cgraph ggml_graph_view(ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ i0, പൂർണ്ണസംഖ്യ i1);

// ggml-alloc.c: true if the operation can reuse memory from its sources
GGML_API bool ggml_op_can_inplace(എണ്ണൽ ggml_op op);


// Memory allocation

GGML_API ശൂന്യം * ggml_aligned_malloc(size_t size);
GGML_API ശൂന്യം ggml_aligned_free(ശൂന്യം * ptr, size_t size);

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

സ്ഥിതം വരിയിൽ ദശാംശം fp32_from_bits(uint32_t w) {
    സംഘം {
        uint32_t as_bits;
        ദശാംശം as_മൂല്യം;
    } fp32;
    fp32.as_bits = w;
    തിരികെ fp32.as_മൂല്യം;
}

സ്ഥിതം വരിയിൽ uint32_t fp32_to_bits(ദശാംശം f) {
    സംഘം {
        ദശാംശം as_മൂല്യം;
        uint32_t as_bits;
    } fp32;
    fp32.as_മൂല്യം = f;
    തിരികെ fp32.as_bits;
}

സ്ഥിതം വരിയിൽ ദശാംശം ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    സ്ഥിരം uint32_t w = (uint32_t) h << 16;
    സ്ഥിരം uint32_t sign = w & UINT32_C(0x80000000);
    സ്ഥിരം uint32_t two_w = w + w;

    സ്ഥിരം uint32_t exp_offset = UINT32_C(0xE0) << 23;
#എങ്കിൽ (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    സ്ഥിരം ദശാംശം exp_scale = 0x1.0p-112f;
#അല്ലെങ്കിൽ
    സ്ഥിരം ദശാംശം exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    സ്ഥിരം ദശാംശം normalized_മൂല്യം = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    സ്ഥിരം uint32_t magic_mask = UINT32_C(126) << 23;
    സ്ഥിരം ദശാംശം magic_bias = 0.5f;
    സ്ഥിരം ദശാംശം denormalized_മൂല്യം = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    സ്ഥിരം uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    സ്ഥിരം uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_മൂല്യം) : fp32_to_bits(normalized_മൂല്യം));
    തിരികെ fp32_from_bits(result);
}

സ്ഥിതം വരിയിൽ ggml_fp16_t ggml_compute_fp32_to_fp16(ദശാംശം f) {
#എങ്കിൽ (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    സ്ഥിരം ദശാംശം scale_to_inf = 0x1.0p+112f;
    സ്ഥിരം ദശാംശം scale_to_zero = 0x1.0p-110f;
#അല്ലെങ്കിൽ
    സ്ഥിരം ദശാംശം scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    സ്ഥിരം ദശാംശം scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    ദശാംശം base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    സ്ഥിരം uint32_t w = fp32_to_bits(f);
    സ്ഥിരം uint32_t shl1_w = w + w;
    സ്ഥിരം uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    എങ്കിൽ (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    സ്ഥിരം uint32_t bits = fp32_to_bits(base);
    സ്ഥിരം uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    സ്ഥിരം uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    സ്ഥിരം uint32_t nonsign = exp_bits + mantissa_bits;
    തിരികെ (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

സ്ഥിതം വരിയിൽ ദശാംശം ggml_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;  // Stores the raw bit representation of the float

    // Handle special case for minimum exponent (denormalized float)
    എങ്കിൽ (x == 0) {
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
    അല്ലെങ്കിൽ {
        // Construct normalized float by shifting exponent into position:
        // - Exponent field: 8 bits (positions 30-23)
        // - Mantissa: 0 (implicit leading 1)
        // Value = 2^(x - 127)
        bits = (uint32_t) x << 23;
    }

    ദശാംശം result;  // Final float value
                   // Safely reinterpret bit pattern as float without type-punning issues
    memcpy(&result, &bits, വലുപ്പം(ദശാംശം));
    തിരികെ result;
}

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
സ്ഥിതം വരിയിൽ ദശാംശം ggml_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;

    // For x < 2: use precomputed denormal patterns
    എങ്കിൽ (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        bits = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    അല്ലെങ്കിൽ {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        bits = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    ദശാംശം result;
    memcpy(&result, &bits, വലുപ്പം(ദശാംശം));
    തിരികെ result;
}

#define GGML_E8M0_TO_FP32(x) ggml_e8m0_to_fp32(x)
#define GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
സ്ഥിതം വരിയിൽ ദശാംശം ggml_ue4m3_to_fp32(uint8_t x) {
    എങ്കിൽ (x == 0 || x == 0x7F) {
        തിരികെ 0.0f;
    }
    പൂർണ്ണസംഖ്യ   exp = (x >> 3) & 0xF;
    പൂർണ്ണസംഖ്യ   man = x & 0x7;
    ദശാംശം raw;
    എങ്കിൽ (exp == 0) {
        raw = ldexpf((ദശാംശം) man, -9);
    } അല്ലെങ്കിൽ {
        raw = ldexpf(1.0f + (ദശാംശം) man / 8.0f, exp - 7);
    }
    തിരികെ raw * 0.5f;
}

സ്ഥിതം വരിയിൽ uint8_t ggml_fp32_to_ue4m3(ദശാംശം x) {
    എങ്കിൽ (!(x > 0.0f)) {
        തിരികെ 0;
    }
    എങ്കിൽ (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t bits;
    memcpy(&bits, &x, 4);
    പൂർണ്ണസംഖ്യ fp32_exp  = ((bits >> 23) & 0xFF) - 127;
    പൂർണ്ണസംഖ്യ fp32_man  = (bits >> 20) & 0x7;
    പൂർണ്ണസംഖ്യ ue4m3_exp = fp32_exp + 7;
    എങ്കിൽ (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        പൂർണ്ണസംഖ്യ man = (പൂർണ്ണസംഖ്യ) (x * 512.0f + 0.5f);
        എങ്കിൽ (man > 7) {
            man = 7;
        }
        എങ്കിൽ (man < 1) {
            തിരികെ 0;
        }
        തിരികെ (uint8_t) man;
    }
    എങ്കിൽ (ue4m3_exp >= 15) {
        തിരികെ 0x7E;
    }
    പൂർണ്ണസംഖ്യ round_bit = (bits >> 19) & 1;
    പൂർണ്ണസംഖ്യ ue4m3_man = fp32_man + round_bit;
    എങ്കിൽ (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        എങ്കിൽ (ue4m3_exp >= 15) {
            തിരികെ 0x7E;
        }
    }
    തിരികെ (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
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
സ്ഥിതം വരിയിൽ ദശാംശം ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    സംഘം {
        ദശാംശം f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    തിരികെ u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
സ്ഥിതം വരിയിൽ ggml_bf16_t ggml_compute_fp32_to_bf16(ദശാംശം s) {
    ggml_bf16_t h;
    സംഘം {
        ദശാംശം f;
        uint32_t i;
    } u;
    u.f = s;
    എങ്കിൽ ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.bits = (u.i >> 16) | 64; /* force to quiet */
        തിരികെ h;
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    തിരികെ h;
}

#define GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)
#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)

സ്ഥിതം വരിയിൽ int32_t ggml_കണ്ണി_get_use_എണ്ണം(സ്ഥിരം ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx) {
    സ്ഥിരം ഘടന ggml_tensor * കണ്ണി = cgraph->nodes[കണ്ണി_idx];

    size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, കണ്ണി);
    എങ്കിൽ (!ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        തിരികെ 0;
    }
    തിരികെ cgraph->use_counts[hash_pos];
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
സ്ഥിതം വരിയിൽ bool ggml_കണ്ണി_has_n_uses(സ്ഥിരം ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx, int32_t n_uses) {
    സ്ഥിരം ഘടന ggml_tensor * കണ്ണി = cgraph->nodes[കണ്ണി_idx];

    // check the use count against how many we're replacing
    എങ്കിൽ (ggml_കണ്ണി_get_use_എണ്ണം(cgraph, കണ്ണി_idx) != n_uses) {
        തിരികെ false;
    }

    // if node is a view, some other node might be using the intermediate result
    // via the view source.
    എങ്കിൽ (കണ്ണി->view_src) {
        തിരികെ false;
    }

    // If the user requested output for the node, can't fuse
    എങ്കിൽ (കണ്ണി->flags & GGML_TENSOR_FLAG_OUTPUT) {
        തിരികെ false;
    }

    തിരികെ true;
}

// Returns true if nodes with indices { node_idxs } are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
സ്ഥിതം വരിയിൽ bool ggml_can_fuse_ext(സ്ഥിരം ഘടന ggml_cgraph * cgraph, സ്ഥിരം പൂർണ്ണസംഖ്യ * കണ്ണി_idxs, സ്ഥിരം എണ്ണൽ ggml_op * ops, പൂർണ്ണസംഖ്യ അക്കം_ops) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < അക്കം_ops; ++i) {
        എങ്കിൽ (കണ്ണി_idxs[i] >= cgraph->n_nodes) {
            തിരികെ false;
        }

        ഘടന ggml_tensor * കണ്ണി = cgraph->nodes[കണ്ണി_idxs[i]];
        എങ്കിൽ (കണ്ണി->op != ops[i]) {
            തിരികെ false;
        }
        എങ്കിൽ ((കണ്ണി->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            തിരികെ false;
        }
        എങ്കിൽ (i < അക്കം_ops - 1 && !ggml_കണ്ണി_has_n_uses(cgraph, കണ്ണി_idxs[i], 1)) {
            തിരികെ false;
        }
        എങ്കിൽ (i > 0) {
            ഘടന ggml_tensor * prev = cgraph->nodes[കണ്ണി_idxs[i - 1]];
            എങ്കിൽ (കണ്ണി->src[0] != prev && കണ്ണി->src[1] != prev) {
                തിരികെ false;
            }
            എങ്കിൽ (!ggml_are_same_shape(കണ്ണി, prev)) {
                തിരികെ false;
            }
        }
    }
    തിരികെ true;
}

// same as above, for sequential indices starting at node_idx
സ്ഥിതം വരിയിൽ bool ggml_can_fuse(സ്ഥിരം ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx, സ്ഥിരം എണ്ണൽ ggml_op * ops, പൂർണ്ണസംഖ്യ അക്കം_ops) {
    പ്രതിജ്ഞാ(അക്കം_ops < 32);

    എങ്കിൽ (കണ്ണി_idx + അക്കം_ops > cgraph->n_nodes) {
        തിരികെ false;
    }

    പൂർണ്ണസംഖ്യ idxs[32];
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < അക്കം_ops; ++i) {
        idxs[i] = കണ്ണി_idx + i;
    }

    തിരികെ ggml_can_fuse_ext(cgraph, idxs, ops, അക്കം_ops);
}

GGML_API bool ggml_can_fuse_subgraph_ext(സ്ഥിരം ഘടന ggml_cgraph * cgraph,
                                         സ്ഥിരം പൂർണ്ണസംഖ്യ *                കണ്ണി_idxs,
                                         പൂർണ്ണസംഖ്യ                        എണ്ണം,
                                         സ്ഥിരം എണ്ണൽ ggml_op *       ops,
                                         സ്ഥിരം പൂർണ്ണസംഖ്യ *                outputs,
                                         പൂർണ്ണസംഖ്യ                        അക്കം_outputs);

// Returns true if the subgraph formed by {node_idxs} can be fused
// checks whethers all nodes which are not part of outputs can be elided
// by checking if their num_uses are confined to the subgraph
സ്ഥിതം വരിയിൽ bool ggml_can_fuse_subgraph(സ്ഥിരം ഘടന ggml_cgraph * cgraph,
                                          പൂർണ്ണസംഖ്യ                        കണ്ണി_idx,
                                          പൂർണ്ണസംഖ്യ                        എണ്ണം,
                                          സ്ഥിരം എണ്ണൽ ggml_op *       ops,
                                          സ്ഥിരം പൂർണ്ണസംഖ്യ *                outputs,
                                          പൂർണ്ണസംഖ്യ                        അക്കം_outputs) {
    GGML_ASSERT(എണ്ണം < 32);
    എങ്കിൽ (കണ്ണി_idx + എണ്ണം > cgraph->n_nodes) {
        തിരികെ false;
    }

    പൂർണ്ണസംഖ്യ idxs[32];

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < എണ്ണം; ++i) {
        idxs[i] = കണ്ണി_idx + i;
    }

    തിരികെ ggml_can_fuse_subgraph_ext(cgraph, idxs, എണ്ണം, ops, outputs, അക്കം_outputs);
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#അന്തര്ഭാവ <array>
#അന്തര്ഭാവ <initializer_list>
#അന്തര്ഭാവ <vector>

// nicer C++ syntax for ggml_can_fuse
വരിയിൽ bool ggml_can_fuse(സ്ഥിരം ഘടന ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx, std::initializer_list<എണ്ണൽ ggml_op> ops) {
    തിരികെ ggml_can_fuse(cgraph, കണ്ണി_idx, ops.begin(), (പൂർണ്ണസംഖ്യ)ops.size());
}

വരിയിൽ bool ggml_can_fuse_subgraph(സ്ഥിരം ഘടന ggml_cgraph *          cgraph,
                                   പൂർണ്ണസംഖ്യ                                 start_idx,
                                   std::initializer_list<എണ്ണൽ ggml_op> ops,
                                   std::initializer_list<പൂർണ്ണസംഖ്യ>          outputs = {}) {
    തിരികെ ggml_can_fuse_subgraph(cgraph, start_idx, ops.size(), ops.begin(), outputs.begin(), outputs.size());
}

// Return true if the edges in the graph match expectations.
വരിയിൽ bool ggml_check_edges(സ്ഥിരം ഘടന ggml_cgraph *                cgraph,
                             പൂർണ്ണസംഖ്യ                                       start_idx,
                             std::initializer_list<std::array<പൂർണ്ണസംഖ്യ, 3>> edges) {
    വേണ്ടി (സ്ഥിരം auto & edge : edges) {
        പൂർണ്ണസംഖ്യ dst_കണ്ണി = edge[0];
        പൂർണ്ണസംഖ്യ src_idx  = edge[1];
        പൂർണ്ണസംഖ്യ src_കണ്ണി = edge[2];
        എങ്കിൽ (cgraph->nodes[start_idx + dst_കണ്ണി]->src[src_idx] != cgraph->nodes[start_idx + src_കണ്ണി]) {
            തിരികെ false;
        }
    }
    തിരികെ true;
}

// expose GGUF internals for test code
GGML_API size_t gguf_type_size(എണ്ണൽ gguf_type type);
GGML_API ശൂന്യം gguf_write_to_buf(സ്ഥിരം ഘടന gguf_context * ctx, std::vector<int8_t> & buf, bool only_meta);
#endif // __cplusplus
