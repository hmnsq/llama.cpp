#आशय once

// GGML internal header

#समावेश "ggml.h"
#समावेश "gguf.h"

#समावेश <assert.h>
#समावेश <math.h>
#समावेश <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#समावेश <stdbool.h>
#समावेश <stdint.h>
#समावेश <string.h>

#अगर_घोषित __ARM_FEATURE_SVE
#समावेश <arm_sve.h>
#पूर्ण_अगर // __ARM_FEATURE_SVE

#अगर defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#समावेश <arm_neon.h>
#पूर्ण_अगर

#अगर_घोषित __cplusplus
बाह्य "C" {
#पूर्ण_अगर

व्योम ggml_print_backtrace(व्योम);

uint64_t ggml_graph_अग्र_uid(व्योम);

#अगर_अघोषित MIN
#    define MIN(a, b) ((a) < (b) ? (a) : (b))
#पूर्ण_अगर

#अगर_अघोषित MAX
#    define MAX(a, b) ((a) > (b) ? (a) : (b))
#पूर्ण_अगर

// required for mmap as gguf only guarantees 32-byte alignment
#घोषणा TENSOR_ALIGNMENT 32

// static_assert should be a #define, but if it's not,
// fall back to the _Static_assert C11 keyword.
// if C99 - static_assert is noop
// ref: https://stackoverflow.com/a/53923785/4039976
#अगर_अघोषित __cplusplus
    #अगर_अघोषित static_assert
        #अगर defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
            #घोषणा static_assert(cond, संदेश) _Static_assert(cond, संदेश)
        #अन्यथा
            #घोषणा static_assert(cond, संदेश) काष्ठा global_scope_noop_trick
        #पूर्ण_अगर
    #पूर्ण_अगर
#पूर्ण_अगर

अटल अंतरभूत पूर्णांक ggml_up32(पूर्णांक n) {
    वापस (n + 31) & ~31;
}

//static inline int ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

अटल अंतरभूत पूर्णांक ggml_up(पूर्णांक n, पूर्णांक m) {
    // assert m is a power of 2
    GGML_ASSERT((m & (m - 1)) == 0);
    वापस (n + m - 1) & ~(m - 1);
}

// TODO: move to ggml.h? (won't be able to inline)
अटल bool ggml_are_same_layout(स्थिर काष्ठा ggml_tensor * a, स्थिर काष्ठा ggml_tensor * b) {
    अगर (a->type != b->type) {
        वापस false;
    }
    क्रम (पूर्णांक i = 0; i < GGML_MAX_DIMS; i++) {
        अगर (a->ne[i] != b->ne[i]) {
            वापस false;
        }
        अगर (a->nb[i] != b->nb[i]) {
            वापस false;
        }
    }
    वापस true;
}

अटल bool ggml_op_is_empty(क्रमागत ggml_op op) {
    चयन (op) {
        हाल GGML_OP_NONE:
        हाल GGML_OP_RESHAPE:
        हाल GGML_OP_TRANSPOSE:
        हाल GGML_OP_VIEW:
        हाल GGML_OP_PERMUTE:
            वापस true;
        शेष:
            वापस false;
    }
}

अटल अंतरभूत bool ggml_impl_is_view(स्थिर काष्ठा ggml_tensor * t) {
    वापस t->view_src != शून्य;
}

अटल अंतरभूत भग्न ggml_compute_softplus_f32(भग्न निवेश) {
    वापस (निवेश > 20.0f) ? निवेश : logf(1 + expf(निवेश));
}
//
// logging
//

GGML_ATTRIBUTE_FORMAT(2, 3)
GGML_API व्योम ggml_log_internal        (क्रमागत ggml_log_level level, स्थिर अक्षर * format, ...);
GGML_API व्योम ggml_log_callback_default(क्रमागत ggml_log_level level, स्थिर अक्षर * text, व्योम * user_तथ्य);

#घोषणा GGML_LOG(...)       ggml_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#घोषणा GGML_LOG_INFO(...)  ggml_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#घोषणा GGML_LOG_WARN(...)  ggml_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#घोषणा GGML_LOG_ERROR(...) ggml_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#घोषणा GGML_LOG_DEBUG(...) ggml_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#घोषणा GGML_LOG_CONT(...)  ggml_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

#घोषणा GGML_DEBUG 0

#अगर (GGML_DEBUG >= 1)
#घोषणा GGML_PRINT_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#अन्यथा
#घोषणा GGML_PRINT_DEBUG(...)
#पूर्ण_अगर

#अगर (GGML_DEBUG >= 5)
#घोषणा GGML_PRINT_DEBUG_5(...) GGML_LOG_DEBUG(__VA_ARGS__)
#अन्यथा
#घोषणा GGML_PRINT_DEBUG_5(...)
#पूर्ण_अगर

#अगर (GGML_DEBUG >= 10)
#घोषणा GGML_PRINT_DEBUG_10(...) GGML_LOG_DEBUG(__VA_ARGS__)
#अन्यथा
#घोषणा GGML_PRINT_DEBUG_10(...)
#पूर्ण_अगर

// tensor params

अटल व्योम ggml_set_op_params(काष्ठा ggml_tensor * tensor, स्थिर व्योम * params, माप_प्रकार params_आकार) {
    GGML_ASSERT(tensor != शून्य); // silence -Warray-bounds warnings
    निश्चित(params_आकार <= GGML_MAX_OP_PARAMS);
    स_नकल(tensor->op_params, params, params_आकार);
}

अटल int32_t ggml_get_op_params_i32(स्थिर काष्ठा ggml_tensor * tensor, uint32_t i) {
    निश्चित(i < GGML_MAX_OP_PARAMS / माप(int32_t));
    वापस ((स्थिर int32_t *)(tensor->op_params))[i];
}

अटल भग्न ggml_get_op_params_f32(स्थिर काष्ठा ggml_tensor * tensor, uint32_t i) {
    निश्चित(i < GGML_MAX_OP_PARAMS / माप(भग्न));
    वापस ((स्थिर भग्न *)(tensor->op_params))[i];
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
अटल व्योम ggml_set_op_params_i32(काष्ठा ggml_tensor * tensor, uint32_t i, int32_t मूल्य) {
    निश्चित(i < GGML_MAX_OP_PARAMS / माप(int32_t));
    ((int32_t *)(tensor->op_params))[i] = मूल्य;
}

अटल व्योम ggml_set_op_params_f32(काष्ठा ggml_tensor * tensor, uint32_t i, भग्न मूल्य) {
    निश्चित(i < GGML_MAX_OP_PARAMS / माप(भग्न));
    ((भग्न *)(tensor->op_params))[i] = मूल्य;
}

काष्ठा ggml_map_custom1_op_params {
    ggml_custom1_op_t  fun;
    पूर्णांक                n_tasks;
    व्योम             * userdata;
};

काष्ठा ggml_map_custom2_op_params {
    ggml_custom2_op_t   fun;
    पूर्णांक                 n_tasks;
    व्योम              * userdata;
};

काष्ठा ggml_map_custom3_op_params {
    ggml_custom3_op_t fun;
    पूर्णांक               n_tasks;
    व्योम            * userdata;
};

काष्ठा ggml_custom_op_params {
    ggml_custom_op_t fun;
    पूर्णांक              n_tasks;
    व्योम           * userdata;
};

// bitset

प्रकार uint32_t ggml_bitset_t;

static_assert(माप(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#घोषणा BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#घोषणा BITSET_MASK (माप(ggml_bitset_t)*8 - 1)

अटल माप_प्रकार ggml_bitset_आकार(माप_प्रकार n) {
    वापस (n + BITSET_MASK) >> BITSET_SHR;
}

अटल अंतरभूत bool ggml_bitset_get(स्थिर ggml_bitset_t * bitset, माप_प्रकार i) {
    वापस !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
}

अटल अंतरभूत व्योम ggml_bitset_set(ggml_bitset_t * bitset, माप_प्रकार i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_MASK));
}

अटल अंतरभूत व्योम ggml_bitset_clear(ggml_bitset_t * bitset, माप_प्रकार i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_MASK));
}

// hash set

#घोषणा GGML_HASHSET_FULL ((माप_प्रकार)-1)
#घोषणा GGML_HASHSET_ALREADY_EXISTS ((माप_प्रकार)-2)

काष्ठा ggml_hash_set {
    माप_प्रकार आकार;
    ggml_bitset_t * used;       // whether or not the keys are in use i.e. set
    काष्ठा ggml_tensor ** कुंजियाँ; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};

काष्ठा ggml_hash_set ggml_hash_set_new(माप_प्रकार आकार);
व्योम                 ggml_hash_set_free(काष्ठा ggml_hash_set * hash_set);

// returns the minimum size for a hash set that can hold min_sz elements
माप_प्रकार ggml_hash_आकार(माप_प्रकार min_sz);

// remove all elements from the hash set
व्योम ggml_hash_set_reset(काष्ठा ggml_hash_set * hash_set);

// returns true if key is in the hash set
अटल bool ggml_hash_contains(स्थिर काष्ठा ggml_hash_set * hash_set, काष्ठा ggml_tensor * कुंजी);

// returns GGML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
अटल माप_प्रकार ggml_hash_find(स्थिर काष्ठा ggml_hash_set * hash_set, स्थिर काष्ठा ggml_tensor * कुंजी);

// returns GGML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
अटल माप_प्रकार ggml_hash_जोड़ो(काष्ठा ggml_hash_set * hash_set, काष्ठा ggml_tensor * कुंजी);

// return index, asserts if table is full
अटल माप_प्रकार ggml_hash_find_or_जोड़ो(काष्ठा ggml_hash_set * hash_set, काष्ठा ggml_tensor * कुंजी);

// hash function for ggml_tensor
अटल अंतरभूत माप_प्रकार ggml_hash(स्थिर काष्ठा ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    वापस (माप_प्रकार)(uintptr_t)p >> 4;
}

अटल माप_प्रकार ggml_hash_find(स्थिर काष्ठा ggml_hash_set * hash_set, स्थिर काष्ठा ggml_tensor * कुंजी) {
    माप_प्रकार h = ggml_hash(कुंजी) % hash_set->आकार;

    // linear probing
    माप_प्रकार i = h;
    जबतक (ggml_bitset_get(hash_set->used, i) && hash_set->कुंजियाँ[i] != कुंजी) {
        i = (i + 1) % hash_set->आकार;
        अगर (i == h) {
            // visited all hash table entries -> not found
            वापस GGML_HASHSET_FULL;
        }
    }
    वापस i;
}

अटल bool ggml_hash_contains(स्थिर काष्ठा ggml_hash_set * hash_set, काष्ठा ggml_tensor * कुंजी) {
    माप_प्रकार i = ggml_hash_find(hash_set, कुंजी);
    वापस i != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, i);
}

अटल माप_प्रकार ggml_hash_जोड़ो(काष्ठा ggml_hash_set * hash_set, काष्ठा ggml_tensor * कुंजी) {
    माप_प्रकार h = ggml_hash(कुंजी) % hash_set->आकार;

    // linear probing
    माप_प्रकार i = h;
    करो {
        अगर (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->कुंजियाँ[i] = कुंजी;
            वापस i;
        }
        अगर (hash_set->कुंजियाँ[i] == कुंजी) {
            वापस GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_set->आकार;
    } जबतक (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

अटल माप_प्रकार ggml_hash_find_or_जोड़ो(काष्ठा ggml_hash_set * hash_set, काष्ठा ggml_tensor * कुंजी) {
    माप_प्रकार h = ggml_hash(कुंजी) % hash_set->आकार;

    // linear probing
    माप_प्रकार i = h;
    करो {
        अगर (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->कुंजियाँ[i] = कुंजी;
            वापस i;
        }
        अगर (hash_set->कुंजियाँ[i] == कुंजी) {
            वापस i;
        }
        i = (i + 1) % hash_set->आकार;
    } जबतक (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

// computation graph

क्रमागत ggml_cgraph_eval_order {
    GGML_CGRAPH_EVAL_ORDER_बायाँ_TO_दायाँ = 0,
    GGML_CGRAPH_EVAL_ORDER_दायाँ_TO_बायाँ,
    GGML_CGRAPH_EVAL_ORDER_गिनती
};

काष्ठा ggml_cgraph {
    पूर्णांक आकार;    // maximum number of nodes/leafs/grads/grad_accs
    पूर्णांक n_nodes; // number of nodes currently in use
    पूर्णांक n_leafs; // number of leafs currently in use

    काष्ठा ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
    काष्ठा ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
    काष्ठा ggml_tensor ** grad_accs; // accumulators for node gradients
    काष्ठा ggml_tensor ** leafs;     // tensors with constant data
    int32_t             * use_counts;// number of uses of each tensor, indexed by hash table slot

    काष्ठा ggml_hash_set visited_hash_set;

    क्रमागत ggml_cgraph_eval_order order;

    // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
    // a value of 0 means it is not set and should be ignored
    uint64_t uid;
};

// returns a slice of cgraph with nodes [i0, i1)
// the slice does not have leafs or gradients
// if you need the gradients, get them from the original graph
काष्ठा ggml_cgraph ggml_graph_view(काष्ठा ggml_cgraph * cgraph, पूर्णांक i0, पूर्णांक i1);

// ggml-alloc.c: true if the operation can reuse memory from its sources
GGML_API bool ggml_op_can_inplace(क्रमागत ggml_op op);


// Memory allocation

GGML_API व्योम * ggml_aligned_malloc(माप_प्रकार आकार);
GGML_API व्योम ggml_aligned_free(व्योम * ptr, माप_प्रकार आकार);

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

अटल अंतरभूत भग्न fp32_from_bits(uint32_t w) {
    जोड़ {
        uint32_t as_bits;
        भग्न as_मूल्य;
    } fp32;
    fp32.as_bits = w;
    वापस fp32.as_मूल्य;
}

अटल अंतरभूत uint32_t fp32_to_bits(भग्न f) {
    जोड़ {
        भग्न as_मूल्य;
        uint32_t as_bits;
    } fp32;
    fp32.as_मूल्य = f;
    वापस fp32.as_bits;
}

अटल अंतरभूत भग्न ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    स्थिर uint32_t w = (uint32_t) h << 16;
    स्थिर uint32_t sign = w & UINT32_C(0x80000000);
    स्थिर uint32_t two_w = w + w;

    स्थिर uint32_t exp_offset = UINT32_C(0xE0) << 23;
#अगर (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    स्थिर भग्न exp_scale = 0x1.0p-112f;
#अन्यथा
    स्थिर भग्न exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#पूर्ण_अगर
    स्थिर भग्न normalized_मूल्य = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    स्थिर uint32_t magic_mask = UINT32_C(126) << 23;
    स्थिर भग्न magic_bias = 0.5f;
    स्थिर भग्न denormalized_मूल्य = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    स्थिर uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    स्थिर uint32_t परिणाम = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_मूल्य) : fp32_to_bits(normalized_मूल्य));
    वापस fp32_from_bits(परिणाम);
}

अटल अंतरभूत ggml_fp16_t ggml_compute_fp32_to_fp16(भग्न f) {
#अगर (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    स्थिर भग्न scale_to_inf = 0x1.0p+112f;
    स्थिर भग्न scale_to_zero = 0x1.0p-110f;
#अन्यथा
    स्थिर भग्न scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    स्थिर भग्न scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#पूर्ण_अगर
    भग्न base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    स्थिर uint32_t w = fp32_to_bits(f);
    स्थिर uint32_t shl1_w = w + w;
    स्थिर uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    अगर (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    स्थिर uint32_t bits = fp32_to_bits(base);
    स्थिर uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    स्थिर uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    स्थिर uint32_t nonsign = exp_bits + mantissa_bits;
    वापस (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#घोषणा GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#घोषणा GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)

#घोषणा GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#घोषणा GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

अटल अंतरभूत भग्न ggml_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;  // Stores the raw bit representation of the float

    // Handle special case for minimum exponent (denormalized float)
    अगर (x == 0) {
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

    भग्न परिणाम;  // Final float value
                   // Safely reinterpret bit pattern as float without type-punning issues
    स_नकल(&परिणाम, &bits, माप(भग्न));
    वापस परिणाम;
}

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
अटल अंतरभूत भग्न ggml_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;

    // For x < 2: use precomputed denormal patterns
    अगर (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        bits = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    अन्यथा {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        bits = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    भग्न परिणाम;
    स_नकल(&परिणाम, &bits, माप(भग्न));
    वापस परिणाम;
}

#घोषणा GGML_E8M0_TO_FP32(x) ggml_e8m0_to_fp32(x)
#घोषणा GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
अटल अंतरभूत भग्न ggml_ue4m3_to_fp32(uint8_t x) {
    अगर (x == 0 || x == 0x7F) {
        वापस 0.0f;
    }
    पूर्णांक   exp = (x >> 3) & 0xF;
    पूर्णांक   man = x & 0x7;
    भग्न raw;
    अगर (exp == 0) {
        raw = ldexpf((भग्न) man, -9);
    } अन्यथा {
        raw = ldexpf(1.0f + (भग्न) man / 8.0f, exp - 7);
    }
    वापस raw * 0.5f;
}

अटल अंतरभूत uint8_t ggml_fp32_to_ue4m3(भग्न x) {
    अगर (!(x > 0.0f)) {
        वापस 0;
    }
    अगर (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t bits;
    स_नकल(&bits, &x, 4);
    पूर्णांक fp32_exp  = ((bits >> 23) & 0xFF) - 127;
    पूर्णांक fp32_man  = (bits >> 20) & 0x7;
    पूर्णांक ue4m3_exp = fp32_exp + 7;
    अगर (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        पूर्णांक man = (पूर्णांक) (x * 512.0f + 0.5f);
        अगर (man > 7) {
            man = 7;
        }
        अगर (man < 1) {
            वापस 0;
        }
        वापस (uint8_t) man;
    }
    अगर (ue4m3_exp >= 15) {
        वापस 0x7E;
    }
    पूर्णांक round_bit = (bits >> 19) & 1;
    पूर्णांक ue4m3_man = fp32_man + round_bit;
    अगर (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        अगर (ue4m3_exp >= 15) {
            वापस 0x7E;
        }
    }
    वापस (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
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
अटल अंतरभूत भग्न ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    जोड़ {
        भग्न f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    वापस u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
अटल अंतरभूत ggml_bf16_t ggml_compute_fp32_to_bf16(भग्न s) {
    ggml_bf16_t h;
    जोड़ {
        भग्न f;
        uint32_t i;
    } u;
    u.f = s;
    अगर ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.bits = (u.i >> 16) | 64; /* force to quiet */
        वापस h;
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    वापस h;
}

#घोषणा GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)
#घोषणा GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)

अटल अंतरभूत int32_t ggml_गाँठ_get_use_गिनती(स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {
    स्थिर काष्ठा ggml_tensor * गाँठ = cgraph->nodes[गाँठ_idx];

    माप_प्रकार hash_pos = ggml_hash_find(&cgraph->visited_hash_set, गाँठ);
    अगर (!ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        वापस 0;
    }
    वापस cgraph->use_counts[hash_pos];
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
अटल अंतरभूत bool ggml_गाँठ_has_n_uses(स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx, int32_t n_uses) {
    स्थिर काष्ठा ggml_tensor * गाँठ = cgraph->nodes[गाँठ_idx];

    // check the use count against how many we're replacing
    अगर (ggml_गाँठ_get_use_गिनती(cgraph, गाँठ_idx) != n_uses) {
        वापस false;
    }

    // if node is a view, some other node might be using the intermediate result
    // via the view source.
    अगर (गाँठ->view_src) {
        वापस false;
    }

    // If the user requested output for the node, can't fuse
    अगर (गाँठ->flags & GGML_TENSOR_FLAG_OUTPUT) {
        वापस false;
    }

    वापस true;
}

// Returns true if nodes with indices { node_idxs } are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
अटल अंतरभूत bool ggml_can_fuse_ext(स्थिर काष्ठा ggml_cgraph * cgraph, स्थिर पूर्णांक * गाँठ_idxs, स्थिर क्रमागत ggml_op * ops, पूर्णांक अंक_ops) {
    क्रम (पूर्णांक i = 0; i < अंक_ops; ++i) {
        अगर (गाँठ_idxs[i] >= cgraph->n_nodes) {
            वापस false;
        }

        काष्ठा ggml_tensor * गाँठ = cgraph->nodes[गाँठ_idxs[i]];
        अगर (गाँठ->op != ops[i]) {
            वापस false;
        }
        अगर ((गाँठ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            वापस false;
        }
        अगर (i < अंक_ops - 1 && !ggml_गाँठ_has_n_uses(cgraph, गाँठ_idxs[i], 1)) {
            वापस false;
        }
        अगर (i > 0) {
            काष्ठा ggml_tensor * पूर्व = cgraph->nodes[गाँठ_idxs[i - 1]];
            अगर (गाँठ->src[0] != पूर्व && गाँठ->src[1] != पूर्व) {
                वापस false;
            }
            अगर (!ggml_are_same_shape(गाँठ, पूर्व)) {
                वापस false;
            }
        }
    }
    वापस true;
}

// same as above, for sequential indices starting at node_idx
अटल अंतरभूत bool ggml_can_fuse(स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx, स्थिर क्रमागत ggml_op * ops, पूर्णांक अंक_ops) {
    निश्चित(अंक_ops < 32);

    अगर (गाँठ_idx + अंक_ops > cgraph->n_nodes) {
        वापस false;
    }

    पूर्णांक idxs[32];
    क्रम (पूर्णांक i = 0; i < अंक_ops; ++i) {
        idxs[i] = गाँठ_idx + i;
    }

    वापस ggml_can_fuse_ext(cgraph, idxs, ops, अंक_ops);
}

GGML_API bool ggml_can_fuse_subgraph_ext(स्थिर काष्ठा ggml_cgraph * cgraph,
                                         स्थिर पूर्णांक *                गाँठ_idxs,
                                         पूर्णांक                        गिनती,
                                         स्थिर क्रमागत ggml_op *       ops,
                                         स्थिर पूर्णांक *                outputs,
                                         पूर्णांक                        अंक_outputs);

// Returns true if the subgraph formed by {node_idxs} can be fused
// checks whethers all nodes which are not part of outputs can be elided
// by checking if their num_uses are confined to the subgraph
अटल अंतरभूत bool ggml_can_fuse_subgraph(स्थिर काष्ठा ggml_cgraph * cgraph,
                                          पूर्णांक                        गाँठ_idx,
                                          पूर्णांक                        गिनती,
                                          स्थिर क्रमागत ggml_op *       ops,
                                          स्थिर पूर्णांक *                outputs,
                                          पूर्णांक                        अंक_outputs) {
    GGML_ASSERT(गिनती < 32);
    अगर (गाँठ_idx + गिनती > cgraph->n_nodes) {
        वापस false;
    }

    पूर्णांक idxs[32];

    क्रम (पूर्णांक i = 0; i < गिनती; ++i) {
        idxs[i] = गाँठ_idx + i;
    }

    वापस ggml_can_fuse_subgraph_ext(cgraph, idxs, गिनती, ops, outputs, अंक_outputs);
}

#अगर_घोषित __cplusplus
}
#पूर्ण_अगर

#अगर_घोषित __cplusplus
#समावेश <array>
#समावेश <initializer_list>
#समावेश <vector>

// nicer C++ syntax for ggml_can_fuse
अंतरभूत bool ggml_can_fuse(स्थिर काष्ठा ggml_cgraph * cgraph, पूर्णांक गाँठ_idx, std::initializer_list<क्रमागत ggml_op> ops) {
    वापस ggml_can_fuse(cgraph, गाँठ_idx, ops.begin(), (पूर्णांक)ops.आकार());
}

अंतरभूत bool ggml_can_fuse_subgraph(स्थिर काष्ठा ggml_cgraph *          cgraph,
                                   पूर्णांक                                 start_idx,
                                   std::initializer_list<क्रमागत ggml_op> ops,
                                   std::initializer_list<पूर्णांक>          outputs = {}) {
    वापस ggml_can_fuse_subgraph(cgraph, start_idx, ops.आकार(), ops.begin(), outputs.begin(), outputs.आकार());
}

// Return true if the edges in the graph match expectations.
अंतरभूत bool ggml_check_edges(स्थिर काष्ठा ggml_cgraph *                cgraph,
                             पूर्णांक                                       start_idx,
                             std::initializer_list<std::array<पूर्णांक, 3>> edges) {
    क्रम (स्थिर स्वतः & edge : edges) {
        पूर्णांक dst_गाँठ = edge[0];
        पूर्णांक src_idx  = edge[1];
        पूर्णांक src_गाँठ = edge[2];
        अगर (cgraph->nodes[start_idx + dst_गाँठ]->src[src_idx] != cgraph->nodes[start_idx + src_गाँठ]) {
            वापस false;
        }
    }
    वापस true;
}

// expose GGUF internals for test code
GGML_API माप_प्रकार gguf_type_आकार(क्रमागत gguf_type type);
GGML_API व्योम gguf_write_to_buf(स्थिर काष्ठा gguf_context * ctx, std::vector<int8_t> & buf, bool only_meta);
#पूर्ण_अगर // __cplusplus
