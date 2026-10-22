#సూచన once

// GGML internal header

#అన్తర్భావ "ggml.h"
#అన్తర్భావ "gguf.h"

#అన్తర్భావ <assert.h>
#అన్తర్భావ <math.h>
#అన్తర్భావ <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <string.h>

#నిర్వచితమైతే __ARM_FEATURE_SVE
#అన్తర్భావ <arm_sve.h>
#అయితే_ముగింపు // __ARM_FEATURE_SVE

#ఒకవేళ defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#అన్తర్భావ <arm_neon.h>
#అయితే_ముగింపు

#నిర్వచితమైతే __cplusplus
బాహ్య "C" {
#అయితే_ముగింపు

శూన్య ggml_print_backtrace(శూన్య);

uint64_t ggml_లేఖాచిత్రం_తదుపరి_uid(శూన్య);

#నిర్వచితం_కాకపోతే MIN
#    define MIN(a, b) ((a) < (b) ? (a) : (b))
#అయితే_ముగింపు

#నిర్వచితం_కాకపోతే MAX
#    define MAX(a, b) ((a) > (b) ? (a) : (b))
#అయితే_ముగింపు

// required for mmap as gguf only guarantees 32-byte alignment
#నిర్వచించు TENSOR_ALIGNMENT 32

// static_assert should be a #define, but if it's not,
// fall back to the _Static_assert C11 keyword.
// if C99 - static_assert is noop
// ref: https://stackoverflow.com/a/53923785/4039976
#నిర్వచితం_కాకపోతే __cplusplus
    #నిర్వచితం_కాకపోతే static_assert
        #ఒకవేళ defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
            #నిర్వచించు static_assert(cond, సందేశం_2) _Static_assert(cond, సందేశం_2)
        #లేకపోతే
            #నిర్వచించు static_assert(cond, సందేశం_2) నిర్మాణం global_scope_noop_trick
        #అయితే_ముగింపు
    #అయితే_ముగింపు
#అయితే_ముగింపు

స్థిర అంతఃవరుస పూర్ణాంకం ggml_up32(పూర్ణాంకం n) {
    తిరిగి (n + 31) & ~31;
}

//static inline int ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

స్థిర అంతఃవరుస పూర్ణాంకం ggml_పైకి(పూర్ణాంకం n, పూర్ణాంకం m) {
    // assert m is a power of 2
    GGML_ASSERT((m & (m - 1)) == 0);
    తిరిగి (n + m - 1) & ~(m - 1);
}

// TODO: move to ggml.h? (won't be able to inline)
స్థిర bool ggml_are_same_layout(స్థిరరాశి నిర్మాణం ggml_tensor * a, స్థిరరాశి నిర్మాణం ggml_tensor * b) {
    ఒకవేళ (a->type != b->type) {
        తిరిగి false;
    }
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; i++) {
        ఒకవేళ (a->ne[i] != b->ne[i]) {
            తిరిగి false;
        }
        ఒకవేళ (a->nb[i] != b->nb[i]) {
            తిరిగి false;
        }
    }
    తిరిగి true;
}

స్థిర bool ggml_op_is_ఖాళీలు(గణన ggml_op op) {
    ఎంపిక (op) {
        స్థితి GGML_OP_NONE:
        స్థితి GGML_OP_RESHAPE:
        స్థితి GGML_OP_TRANSPOSE:
        స్థితి GGML_OP_VIEW:
        స్థితి GGML_OP_క్రమమార్పు:
            తిరిగి true;
        అప్రమేయ:
            తిరిగి false;
    }
}

స్థిర అంతఃవరుస bool ggml_impl_is_view(స్థిరరాశి నిర్మాణం ggml_tensor * t) {
    తిరిగి t->view_src != ఖాళీ;
}

స్థిర అంతఃవరుస దశాంశం ggml_compute_softplus_f32(దశాంశం ఆదానం) {
    తిరిగి (ఆదానం > 20.0f) ? ఆదానం : logf(1 + expf(ఆదానం));
}
//
// logging
//

GGML_ATTRIBUTE_FORMAT(2, 3)
GGML_API శూన్య ggml_log_అంతర్గత        (గణన ggml_log_level level, స్థిరరాశి అక్షరం * format, ...);
GGML_API శూన్య ggml_log_callback_default(గణన ggml_log_level level, స్థిరరాశి అక్షరం * పాఠ్యం_2, శూన్య * user_దత్తాంశం);

#నిర్వచించు GGML_LOG(...)       ggml_log_అంతర్గత(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#నిర్వచించు GGML_LOG_INFO(...)  ggml_log_అంతర్గత(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#నిర్వచించు GGML_LOG_WARN(...)  ggml_log_అంతర్గత(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#నిర్వచించు GGML_LOG_ERROR(...) ggml_log_అంతర్గత(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#నిర్వచించు GGML_LOG_DEBUG(...) ggml_log_అంతర్గత(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#నిర్వచించు GGML_LOG_CONT(...)  ggml_log_అంతర్గత(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

#నిర్వచించు GGML_DEBUG 0

#ఒకవేళ (GGML_DEBUG >= 1)
#నిర్వచించు GGML_PRINT_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)
#లేకపోతే
#నిర్వచించు GGML_PRINT_DEBUG(...)
#అయితే_ముగింపు

#ఒకవేళ (GGML_DEBUG >= 5)
#నిర్వచించు GGML_PRINT_DEBUG_5(...) GGML_LOG_DEBUG(__VA_ARGS__)
#లేకపోతే
#నిర్వచించు GGML_PRINT_DEBUG_5(...)
#అయితే_ముగింపు

#ఒకవేళ (GGML_DEBUG >= 10)
#నిర్వచించు GGML_PRINT_DEBUG_10(...) GGML_LOG_DEBUG(__VA_ARGS__)
#లేకపోతే
#నిర్వచించు GGML_PRINT_DEBUG_10(...)
#అయితే_ముగింపు

// tensor params

స్థిర శూన్య ggml_సమితి_op_params(నిర్మాణం ggml_tensor * tensor, స్థిరరాశి శూన్య * params, size_t params_కొలత) {
    GGML_ASSERT(tensor != ఖాళీ); // silence -Warray-bounds warnings
    ప్రతిజ్ఞా(params_కొలత <= GGML_MAX_OP_PARAMS);
    జ్ఞాపకం_నకలు(tensor->op_params, params, params_కొలత);
}

స్థిర int32_t ggml_get_op_params_i32(స్థిరరాశి నిర్మాణం ggml_tensor * tensor, uint32_t i) {
    ప్రతిజ్ఞా(i < GGML_MAX_OP_PARAMS / పరిమాణం(int32_t));
    తిరిగి ((స్థిరరాశి int32_t *)(tensor->op_params))[i];
}

స్థిర దశాంశం ggml_get_op_params_f32(స్థిరరాశి నిర్మాణం ggml_tensor * tensor, uint32_t i) {
    ప్రతిజ్ఞా(i < GGML_MAX_OP_PARAMS / పరిమాణం(దశాంశం));
    తిరిగి ((స్థిరరాశి దశాంశం *)(tensor->op_params))[i];
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
స్థిర శూన్య ggml_సమితి_op_params_i32(నిర్మాణం ggml_tensor * tensor, uint32_t i, int32_t విలువ) {
    ప్రతిజ్ఞా(i < GGML_MAX_OP_PARAMS / పరిమాణం(int32_t));
    ((int32_t *)(tensor->op_params))[i] = విలువ;
}

స్థిర శూన్య ggml_సమితి_op_params_f32(నిర్మాణం ggml_tensor * tensor, uint32_t i, దశాంశం విలువ) {
    ప్రతిజ్ఞా(i < GGML_MAX_OP_PARAMS / పరిమాణం(దశాంశం));
    ((దశాంశం *)(tensor->op_params))[i] = విలువ;
}

నిర్మాణం ggml_map_custom1_op_params {
    ggml_custom1_op_t  fun;
    పూర్ణాంకం                n_tasks;
    శూన్య             * userdata;
};

నిర్మాణం ggml_map_custom2_op_params {
    ggml_custom2_op_t   fun;
    పూర్ణాంకం                 n_tasks;
    శూన్య              * userdata;
};

నిర్మాణం ggml_map_custom3_op_params {
    ggml_custom3_op_t fun;
    పూర్ణాంకం               n_tasks;
    శూన్య            * userdata;
};

నిర్మాణం ggml_custom_op_params {
    ggml_custom_op_t fun;
    పూర్ణాంకం              n_tasks;
    శూన్య           * userdata;
};

// bitset

రకనిర్వచనం uint32_t ggml_bitset_t;

static_assert(పరిమాణం(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#నిర్వచించు BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)
#నిర్వచించు BITSET_ముసుగు (పరిమాణం(ggml_bitset_t)*8 - 1)

స్థిర size_t ggml_bitset_కొలత(size_t n) {
    తిరిగి (n + BITSET_ముసుగు) >> BITSET_SHR;
}

స్థిర అంతఃవరుస bool ggml_bitset_get(స్థిరరాశి ggml_bitset_t * bitset, size_t i) {
    తిరిగి !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_ముసుగు)));
}

స్థిర అంతఃవరుస శూన్య ggml_bitset_సమితి(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_ముసుగు));
}

స్థిర అంతఃవరుస శూన్య ggml_bitset_clear(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_ముసుగు));
}

// hash set

#నిర్వచించు GGML_HASHSET_నిండిన ((size_t)-1)
#నిర్వచించు GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)

నిర్మాణం ggml_hash_సమితి {
    size_t కొలత;
    ggml_bitset_t * వాడిన;       // whether or not the keys are in use i.e. set
    నిర్మాణం ggml_tensor ** కీలకాలు; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};

నిర్మాణం ggml_hash_సమితి ggml_hash_సమితి_new(size_t కొలత);
శూన్య                 ggml_hash_సమితి_free(నిర్మాణం ggml_hash_సమితి * hash_సమితి);

// returns the minimum size for a hash set that can hold min_sz elements
size_t ggml_hash_కొలత(size_t min_sz);

// remove all elements from the hash set
శూన్య ggml_hash_సమితి_reset(నిర్మాణం ggml_hash_సమితి * hash_సమితి);

// returns true if key is in the hash set
స్థిర bool ggml_hash_కలిగియుంది(స్థిరరాశి నిర్మాణం ggml_hash_సమితి * hash_సమితి, నిర్మాణం ggml_tensor * కీలకం);

// returns GGML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
స్థిర size_t ggml_hash_కనుగొను(స్థిరరాశి నిర్మాణం ggml_hash_సమితి * hash_సమితి, స్థిరరాశి నిర్మాణం ggml_tensor * కీలకం);

// returns GGML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
స్థిర size_t ggml_hash_చొప్పించు(నిర్మాణం ggml_hash_సమితి * hash_సమితి, నిర్మాణం ggml_tensor * కీలకం);

// return index, asserts if table is full
స్థిర size_t ggml_hash_కనుగొను_or_చొప్పించు(నిర్మాణం ggml_hash_సమితి * hash_సమితి, నిర్మాణం ggml_tensor * కీలకం);

// hash function for ggml_tensor
స్థిర అంతఃవరుస size_t ggml_hash(స్థిరరాశి నిర్మాణం ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    తిరిగి (size_t)(uintptr_t)p >> 4;
}

స్థిర size_t ggml_hash_కనుగొను(స్థిరరాశి నిర్మాణం ggml_hash_సమితి * hash_సమితి, స్థిరరాశి నిర్మాణం ggml_tensor * కీలకం) {
    size_t h = ggml_hash(కీలకం) % hash_సమితి->కొలత;

    // linear probing
    size_t i = h;
    అయితే (ggml_bitset_get(hash_సమితి->వాడిన, i) && hash_సమితి->కీలకాలు[i] != కీలకం) {
        i = (i + 1) % hash_సమితి->కొలత;
        ఒకవేళ (i == h) {
            // visited all hash table entries -> not found
            తిరిగి GGML_HASHSET_నిండిన;
        }
    }
    తిరిగి i;
}

స్థిర bool ggml_hash_కలిగియుంది(స్థిరరాశి నిర్మాణం ggml_hash_సమితి * hash_సమితి, నిర్మాణం ggml_tensor * కీలకం) {
    size_t i = ggml_hash_కనుగొను(hash_సమితి, కీలకం);
    తిరిగి i != GGML_HASHSET_నిండిన && ggml_bitset_get(hash_సమితి->వాడిన, i);
}

స్థిర size_t ggml_hash_చొప్పించు(నిర్మాణం ggml_hash_సమితి * hash_సమితి, నిర్మాణం ggml_tensor * కీలకం) {
    size_t h = ggml_hash(కీలకం) % hash_సమితి->కొలత;

    // linear probing
    size_t i = h;
    చేయి {
        ఒకవేళ (!ggml_bitset_get(hash_సమితి->వాడిన, i)) {
            ggml_bitset_సమితి(hash_సమితి->వాడిన, i);
            hash_సమితి->కీలకాలు[i] = కీలకం;
            తిరిగి i;
        }
        ఒకవేళ (hash_సమితి->కీలకాలు[i] == కీలకం) {
            తిరిగి GGML_HASHSET_ALREADY_EXISTS;
        }
        i = (i + 1) % hash_సమితి->కొలత;
    } అయితే (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

స్థిర size_t ggml_hash_కనుగొను_or_చొప్పించు(నిర్మాణం ggml_hash_సమితి * hash_సమితి, నిర్మాణం ggml_tensor * కీలకం) {
    size_t h = ggml_hash(కీలకం) % hash_సమితి->కొలత;

    // linear probing
    size_t i = h;
    చేయి {
        ఒకవేళ (!ggml_bitset_get(hash_సమితి->వాడిన, i)) {
            ggml_bitset_సమితి(hash_సమితి->వాడిన, i);
            hash_సమితి->కీలకాలు[i] = కీలకం;
            తిరిగి i;
        }
        ఒకవేళ (hash_సమితి->కీలకాలు[i] == కీలకం) {
            తిరిగి i;
        }
        i = (i + 1) % hash_సమితి->కొలత;
    } అయితే (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

// computation graph

గణన ggml_cgraph_eval_క్రమం {
    GGML_CGRAPH_EVAL_క్రమం_ఎడమ_గమ్యం_కుడి = 0,
    GGML_CGRAPH_EVAL_క్రమం_కుడి_గమ్యం_ఎడమ,
    GGML_CGRAPH_EVAL_క్రమం_లెక్క
};

నిర్మాణం ggml_cgraph {
    పూర్ణాంకం కొలత;    // maximum number of nodes/leafs/grads/grad_accs
    పూర్ణాంకం n_nodes; // number of nodes currently in use
    పూర్ణాంకం n_leafs; // number of leafs currently in use

    నిర్మాణం ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
    నిర్మాణం ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
    నిర్మాణం ggml_tensor ** grad_accs; // accumulators for node gradients
    నిర్మాణం ggml_tensor ** leafs;     // tensors with constant data
    int32_t             * use_counts;// number of uses of each tensor, indexed by hash table slot

    నిర్మాణం ggml_hash_సమితి సందర్శించిన_hash_సమితి;

    గణన ggml_cgraph_eval_క్రమం క్రమం;

    // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
    // a value of 0 means it is not set and should be ignored
    uint64_t uid;
};

// returns a slice of cgraph with nodes [i0, i1)
// the slice does not have leafs or gradients
// if you need the gradients, get them from the original graph
నిర్మాణం ggml_cgraph ggml_లేఖాచిత్రం_view(నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం i0, పూర్ణాంకం i1);

// ggml-alloc.c: true if the operation can reuse memory from its sources
GGML_API bool ggml_op_can_inplace(గణన ggml_op op);


// Memory allocation

GGML_API శూన్య * ggml_aligned_malloc(size_t కొలత);
GGML_API శూన్య ggml_aligned_free(శూన్య * ptr, size_t కొలత);

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

స్థిర అంతఃవరుస దశాంశం fp32_from_బిట్లు(uint32_t w) {
    సంఘం {
        uint32_t as_బిట్లు;
        దశాంశం as_విలువ;
    } fp32;
    fp32.as_బిట్లు = w;
    తిరిగి fp32.as_విలువ;
}

స్థిర అంతఃవరుస uint32_t fp32_గమ్యం_బిట్లు(దశాంశం f) {
    సంఘం {
        దశాంశం as_విలువ;
        uint32_t as_బిట్లు;
    } fp32;
    fp32.as_విలువ = f;
    తిరిగి fp32.as_బిట్లు;
}

స్థిర అంతఃవరుస దశాంశం ggml_compute_fp16_గమ్యం_fp32(ggml_fp16_t h) {
    స్థిరరాశి uint32_t w = (uint32_t) h << 16;
    స్థిరరాశి uint32_t sign = w & UINT32_C(0x80000000);
    స్థిరరాశి uint32_t two_w = w + w;

    స్థిరరాశి uint32_t exp_స్థానభ్రంశం = UINT32_C(0xE0) << 23;
#ఒకవేళ (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    స్థిరరాశి దశాంశం exp_scale = 0x1.0p-112f;
#లేకపోతే
    స్థిరరాశి దశాంశం exp_scale = fp32_from_బిట్లు(UINT32_C(0x7800000));
#అయితే_ముగింపు
    స్థిరరాశి దశాంశం normalized_విలువ = fp32_from_బిట్లు((two_w >> 4) + exp_స్థానభ్రంశం) * exp_scale;

    స్థిరరాశి uint32_t magic_ముసుగు = UINT32_C(126) << 23;
    స్థిరరాశి దశాంశం magic_bias = 0.5f;
    స్థిరరాశి దశాంశం denormalized_విలువ = fp32_from_బిట్లు((two_w >> 17) | magic_ముసుగు) - magic_bias;

    స్థిరరాశి uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    స్థిరరాశి uint32_t ఫలితం = sign |
        (two_w < denormalized_cutoff ? fp32_గమ్యం_బిట్లు(denormalized_విలువ) : fp32_గమ్యం_బిట్లు(normalized_విలువ));
    తిరిగి fp32_from_బిట్లు(ఫలితం);
}

స్థిర అంతఃవరుస ggml_fp16_t ggml_compute_fp32_గమ్యం_fp16(దశాంశం f) {
#ఒకవేళ (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    స్థిరరాశి దశాంశం scale_గమ్యం_inf = 0x1.0p+112f;
    స్థిరరాశి దశాంశం scale_గమ్యం_zero = 0x1.0p-110f;
#లేకపోతే
    స్థిరరాశి దశాంశం scale_గమ్యం_inf = fp32_from_బిట్లు(UINT32_C(0x77800000));
    స్థిరరాశి దశాంశం scale_గమ్యం_zero = fp32_from_బిట్లు(UINT32_C(0x08800000));
#అయితే_ముగింపు
    దశాంశం పునాది = (fabsf(f) * scale_గమ్యం_inf) * scale_గమ్యం_zero;

    స్థిరరాశి uint32_t w = fp32_గమ్యం_బిట్లు(f);
    స్థిరరాశి uint32_t shl1_w = w + w;
    స్థిరరాశి uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    ఒకవేళ (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    పునాది = fp32_from_బిట్లు((bias >> 1) + UINT32_C(0x07800000)) + పునాది;
    స్థిరరాశి uint32_t బిట్లు = fp32_గమ్యం_బిట్లు(పునాది);
    స్థిరరాశి uint32_t exp_బిట్లు = (బిట్లు >> 13) & UINT32_C(0x00007C00);
    స్థిరరాశి uint32_t mantissa_బిట్లు = బిట్లు & UINT32_C(0x00000FFF);
    స్థిరరాశి uint32_t nonsign = exp_బిట్లు + mantissa_బిట్లు;
    తిరిగి (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#నిర్వచించు GGML_COMPUTE_FP16_గమ్యం_FP32(x) ggml_compute_fp16_గమ్యం_fp32(x)
#నిర్వచించు GGML_COMPUTE_FP32_గమ్యం_FP16(x) ggml_compute_fp32_గమ్యం_fp16(x)

#నిర్వచించు GGML_FP16_గమ్యం_FP32(x) GGML_COMPUTE_FP16_గమ్యం_FP32(x)
#నిర్వచించు GGML_FP32_గమ్యం_FP16(x) GGML_COMPUTE_FP32_గమ్యం_FP16(x)

స్థిర అంతఃవరుస దశాంశం ggml_e8m0_గమ్యం_fp32(uint8_t x) {
    uint32_t బిట్లు;  // Stores the raw bit representation of the float

    // Handle special case for minimum exponent (denormalized float)
    ఒకవేళ (x == 0) {
        // Bit pattern for 2^(-127):
        // - Sign bit: 0 (positive)
        // - Exponent: 0 (denormalized number)
        // - Mantissa: 0x400000 (0.5 in fractional form)
        // Value = 0.5 * 2^(-126) = 2^(-127)
        బిట్లు = 0x00400000;
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
    లేకపోతే {
        // Construct normalized float by shifting exponent into position:
        // - Exponent field: 8 bits (positions 30-23)
        // - Mantissa: 0 (implicit leading 1)
        // Value = 2^(x - 127)
        బిట్లు = (uint32_t) x << 23;
    }

    దశాంశం ఫలితం;  // Final float value
                   // Safely reinterpret bit pattern as float without type-punning issues
    జ్ఞాపకం_నకలు(&ఫలితం, &బిట్లు, పరిమాణం(దశాంశం));
    తిరిగి ఫలితం;
}

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
స్థిర అంతఃవరుస దశాంశం ggml_e8m0_గమ్యం_fp32_half(uint8_t x) {
    uint32_t బిట్లు;

    // For x < 2: use precomputed denormal patterns
    ఒకవేళ (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        బిట్లు = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    లేకపోతే {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        బిట్లు = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    దశాంశం ఫలితం;
    జ్ఞాపకం_నకలు(&ఫలితం, &బిట్లు, పరిమాణం(దశాంశం));
    తిరిగి ఫలితం;
}

#నిర్వచించు GGML_E8_M0_గమ్యం_FP32(x) ggml_e8m0_గమ్యం_fp32(x)
#నిర్వచించు GGML_E8_M0_గమ్యం_FP32_HALF(x) ggml_e8m0_గమ్యం_fp32_half(x)

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
స్థిర అంతఃవరుస దశాంశం ggml_ue4m3_గమ్యం_fp32(uint8_t x) {
    ఒకవేళ (x == 0 || x == 0x7F) {
        తిరిగి 0.0f;
    }
    పూర్ణాంకం   ఘాతాంకం = (x >> 3) & 0xF;
    పూర్ణాంకం   man = x & 0x7;
    దశాంశం raw;
    ఒకవేళ (ఘాతాంకం == 0) {
        raw = ldexpf((దశాంశం) man, -9);
    } లేకపోతే {
        raw = ldexpf(1.0f + (దశాంశం) man / 8.0f, ఘాతాంకం - 7);
    }
    తిరిగి raw * 0.5f;
}

స్థిర అంతఃవరుస uint8_t ggml_fp32_గమ్యం_ue4m3(దశాంశం x) {
    ఒకవేళ (!(x > 0.0f)) {
        తిరిగి 0;
    }
    ఒకవేళ (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t బిట్లు;
    జ్ఞాపకం_నకలు(&బిట్లు, &x, 4);
    పూర్ణాంకం fp32_exp  = ((బిట్లు >> 23) & 0xFF) - 127;
    పూర్ణాంకం fp32_man  = (బిట్లు >> 20) & 0x7;
    పూర్ణాంకం ue4m3_exp = fp32_exp + 7;
    ఒకవేళ (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        పూర్ణాంకం man = (పూర్ణాంకం) (x * 512.0f + 0.5f);
        ఒకవేళ (man > 7) {
            man = 7;
        }
        ఒకవేళ (man < 1) {
            తిరిగి 0;
        }
        తిరిగి (uint8_t) man;
    }
    ఒకవేళ (ue4m3_exp >= 15) {
        తిరిగి 0x7E;
    }
    పూర్ణాంకం విడత_బిట్టు = (బిట్లు >> 19) & 1;
    పూర్ణాంకం ue4m3_man = fp32_man + విడత_బిట్టు;
    ఒకవేళ (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        ఒకవేళ (ue4m3_exp >= 15) {
            తిరిగి 0x7E;
        }
    }
    తిరిగి (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
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
స్థిర అంతఃవరుస దశాంశం ggml_compute_bf16_గమ్యం_fp32(ggml_bf16_t h) {
    సంఘం {
        దశాంశం f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.బిట్లు << 16;
    తిరిగి u.f;
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
స్థిర అంతఃవరుస ggml_bf16_t ggml_compute_fp32_గమ్యం_bf16(దశాంశం s) {
    ggml_bf16_t h;
    సంఘం {
        దశాంశం f;
        uint32_t i;
    } u;
    u.f = s;
    ఒకవేళ ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.బిట్లు = (u.i >> 16) | 64; /* force to quiet */
        తిరిగి h;
    }
    h.బిట్లు = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    తిరిగి h;
}

#నిర్వచించు GGML_FP32_గమ్యం_BF16(x) ggml_compute_fp32_గమ్యం_bf16(x)
#నిర్వచించు GGML_BF16_గమ్యం_FP32(x) ggml_compute_bf16_గమ్యం_fp32(x)

స్థిర అంతఃవరుస int32_t ggml_కణుపు_get_use_లెక్క(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {
    స్థిరరాశి నిర్మాణం ggml_tensor * కణుపు = cgraph->nodes[కణుపు_idx];

    size_t hash_pos = ggml_hash_కనుగొను(&cgraph->సందర్శించిన_hash_సమితి, కణుపు);
    ఒకవేళ (!ggml_bitset_get(cgraph->సందర్శించిన_hash_సమితి.వాడిన, hash_pos)) {
        తిరిగి 0;
    }
    తిరిగి cgraph->use_counts[hash_pos];
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
స్థిర అంతఃవరుస bool ggml_కణుపు_has_n_ఉపయోగిస్తుంది(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, int32_t n_ఉపయోగిస్తుంది) {
    స్థిరరాశి నిర్మాణం ggml_tensor * కణుపు = cgraph->nodes[కణుపు_idx];

    // check the use count against how many we're replacing
    ఒకవేళ (ggml_కణుపు_get_use_లెక్క(cgraph, కణుపు_idx) != n_ఉపయోగిస్తుంది) {
        తిరిగి false;
    }

    // if node is a view, some other node might be using the intermediate result
    // via the view source.
    ఒకవేళ (కణుపు->view_src) {
        తిరిగి false;
    }

    // If the user requested output for the node, can't fuse
    ఒకవేళ (కణుపు->flags & GGML_TENSOR_FLAG_OUTPUT) {
        తిరిగి false;
    }

    తిరిగి true;
}

// Returns true if nodes with indices { node_idxs } are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
స్థిర అంతఃవరుస bool ggml_can_fuse_ext(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, స్థిరరాశి పూర్ణాంకం * కణుపు_idxs, స్థిరరాశి గణన ggml_op * ops, పూర్ణాంకం అంకె_ops) {
    కొరకు (పూర్ణాంకం i = 0; i < అంకె_ops; ++i) {
        ఒకవేళ (కణుపు_idxs[i] >= cgraph->n_nodes) {
            తిరిగి false;
        }

        నిర్మాణం ggml_tensor * కణుపు = cgraph->nodes[కణుపు_idxs[i]];
        ఒకవేళ (కణుపు->op != ops[i]) {
            తిరిగి false;
        }
        ఒకవేళ ((కణుపు->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            తిరిగి false;
        }
        ఒకవేళ (i < అంకె_ops - 1 && !ggml_కణుపు_has_n_ఉపయోగిస్తుంది(cgraph, కణుపు_idxs[i], 1)) {
            తిరిగి false;
        }
        ఒకవేళ (i > 0) {
            నిర్మాణం ggml_tensor * prev = cgraph->nodes[కణుపు_idxs[i - 1]];
            ఒకవేళ (కణుపు->src[0] != prev && కణుపు->src[1] != prev) {
                తిరిగి false;
            }
            ఒకవేళ (!ggml_are_same_shape(కణుపు, prev)) {
                తిరిగి false;
            }
        }
    }
    తిరిగి true;
}

// same as above, for sequential indices starting at node_idx
స్థిర అంతఃవరుస bool ggml_can_fuse(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, స్థిరరాశి గణన ggml_op * ops, పూర్ణాంకం అంకె_ops) {
    ప్రతిజ్ఞా(అంకె_ops < 32);

    ఒకవేళ (కణుపు_idx + అంకె_ops > cgraph->n_nodes) {
        తిరిగి false;
    }

    పూర్ణాంకం idxs[32];
    కొరకు (పూర్ణాంకం i = 0; i < అంకె_ops; ++i) {
        idxs[i] = కణుపు_idx + i;
    }

    తిరిగి ggml_can_fuse_ext(cgraph, idxs, ops, అంకె_ops);
}

GGML_API bool ggml_can_fuse_subgraph_ext(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph,
                                         స్థిరరాశి పూర్ణాంకం *                కణుపు_idxs,
                                         పూర్ణాంకం                        లెక్క,
                                         స్థిరరాశి గణన ggml_op *       ops,
                                         స్థిరరాశి పూర్ణాంకం *                outputs,
                                         పూర్ణాంకం                        అంకె_outputs);

// Returns true if the subgraph formed by {node_idxs} can be fused
// checks whethers all nodes which are not part of outputs can be elided
// by checking if their num_uses are confined to the subgraph
స్థిర అంతఃవరుస bool ggml_can_fuse_subgraph(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph,
                                          పూర్ణాంకం                        కణుపు_idx,
                                          పూర్ణాంకం                        లెక్క,
                                          స్థిరరాశి గణన ggml_op *       ops,
                                          స్థిరరాశి పూర్ణాంకం *                outputs,
                                          పూర్ణాంకం                        అంకె_outputs) {
    GGML_ASSERT(లెక్క < 32);
    ఒకవేళ (కణుపు_idx + లెక్క > cgraph->n_nodes) {
        తిరిగి false;
    }

    పూర్ణాంకం idxs[32];

    కొరకు (పూర్ణాంకం i = 0; i < లెక్క; ++i) {
        idxs[i] = కణుపు_idx + i;
    }

    తిరిగి ggml_can_fuse_subgraph_ext(cgraph, idxs, లెక్క, ops, outputs, అంకె_outputs);
}

#నిర్వచితమైతే __cplusplus
}
#అయితే_ముగింపు

#నిర్వచితమైతే __cplusplus
#అన్తర్భావ <array>
#అన్తర్భావ <initializer_list>
#అన్తర్భావ <vector>

// nicer C++ syntax for ggml_can_fuse
అంతఃవరుస bool ggml_can_fuse(స్థిరరాశి నిర్మాణం ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, std::initializer_list<గణన ggml_op> ops) {
    తిరిగి ggml_can_fuse(cgraph, కణుపు_idx, ops.begin(), (పూర్ణాంకం)ops.కొలత());
}

అంతఃవరుస bool ggml_can_fuse_subgraph(స్థిరరాశి నిర్మాణం ggml_cgraph *          cgraph,
                                   పూర్ణాంకం                                 ప్రారంభం_idx,
                                   std::initializer_list<గణన ggml_op> ops,
                                   std::initializer_list<పూర్ణాంకం>          outputs = {}) {
    తిరిగి ggml_can_fuse_subgraph(cgraph, ప్రారంభం_idx, ops.కొలత(), ops.begin(), outputs.begin(), outputs.కొలత());
}

// Return true if the edges in the graph match expectations.
అంతఃవరుస bool ggml_check_అంచులు(స్థిరరాశి నిర్మాణం ggml_cgraph *                cgraph,
                             పూర్ణాంకం                                       ప్రారంభం_idx,
                             std::initializer_list<std::array<పూర్ణాంకం, 3>> అంచులు) {
    కొరకు (స్థిరరాశి స్వయంచాలక & అంచు : అంచులు) {
        పూర్ణాంకం dst_కణుపు = అంచు[0];
        పూర్ణాంకం src_idx  = అంచు[1];
        పూర్ణాంకం src_కణుపు = అంచు[2];
        ఒకవేళ (cgraph->nodes[ప్రారంభం_idx + dst_కణుపు]->src[src_idx] != cgraph->nodes[ప్రారంభం_idx + src_కణుపు]) {
            తిరిగి false;
        }
    }
    తిరిగి true;
}

// expose GGUF internals for test code
GGML_API size_t gguf_type_కొలత(గణన gguf_type type);
GGML_API శూన్య gguf_write_గమ్యం_buf(స్థిరరాశి నిర్మాణం gguf_context * ctx, std::సదిశ<int8_t> & buf, bool only_meta);
#అయితే_ముగింపు // __cplusplus
