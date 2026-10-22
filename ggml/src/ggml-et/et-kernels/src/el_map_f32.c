// Element-wise operations: dst[i] = src0[i] op src1[i]
#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

// Generic m0-gated element-wise block operation.
// The OP parameter selects the instruction: "fmul.ps", "fadd.ps", "fsub.ps".
#ವ್ಯಾಖ್ಯಾನಿಸು DEFINE_ಖಂಡ_OP(name, op_insn)                                                                             \
    ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ name(ದಶಮಾಂಶ * dst_ಖಂಡ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_ಖಂಡ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src1_ಖಂಡ, ಪೂರ್ಣಾಂಕ elements) { \
        ಸ್ಥಿರಾಂಕ int32_t vec_ಕೊನೆ = (elements / 8) * 8;                                                                \
        ಸ್ಥಿರಾಂಕ int32_t ಬಾಲ    = elements - vec_ಕೊನೆ;                                                                \
                                                                                                                   \
        ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ;                                                                                   \
        __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));                                                         \
        __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");                                                                  \
                                                                                                                   \
        ಗಾಗಿ (int32_t i = 0; i < vec_ಕೊನೆ; i += 8) {                                                                 \
            __asm__ ಅಸ್ಥಿರ(                                                                                      \
                "flw.ps f10, %[s0]\n"                                                                              \
                "flw.ps f11, %[s1]\n" op_insn                                                                      \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, %[d]\n"                                                                               \
                : [d] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ಖಂಡ[i])                                                         \
                : [s0] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src0_ಖಂಡ[i]), [s1] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src1_ಖಂಡ[i]) \
                : "f10", "f11", "f12");                                                                            \
        }                                                                                                          \
        /* Deal with tail chunks */                                                                                \
        ಆದರೆ (ಬಾಲ > 0) {                                                                                            \
            ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ಬಾಲ_m0 = (1ul << ಬಾಲ) - 1;                                                       \
            __asm__ ಅಸ್ಥಿರ(                                                                                      \
                "mov.m.x m0, %[tm], 0\n"                                                                           \
                "flw.ps f10, 0(%[s0])\n"                                                                           \
                "flw.ps f11, 0(%[s1])\n" op_insn                                                                   \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, 0(%[d])\n"                                                                            \
                :                                                                                                  \
                : [s0] "r"(&src0_ಖಂಡ[vec_ಕೊನೆ]), [s1] "r"(&src1_ಖಂಡ[vec_ಕೊನೆ]), [d] "r"(&dst_ಖಂಡ[vec_ಕೊನೆ]),    \
                  [tm] "r"(ಬಾಲ_m0)                                                                                \
                : "f10", "f11", "f12", "memory");                                                                  \
        }                                                                                                          \
                                                                                                                   \
        __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));                                                          \
    }

DEFINE_ಖಂಡ_OP(ಖಂಡ_mul_cache_aligned, "fmul.ps")
DEFINE_ಖಂಡ_OP(ಖಂಡ_ಕೂಡಿಸು_cache_aligned, "fadd.ps")
DEFINE_ಖಂಡ_OP(ಖಂಡ_sub_cache_aligned, "fsub.ps")

// Broadcast variants: src1 is a single scalar, broadcast to all 8 lanes.
#ವ್ಯಾಖ್ಯಾನಿಸು DEFINE_ಖಂಡ_OP_ಪ್ರಸಾರ(name, op_insn)                                                                     \
    ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ name(ದಶಮಾಂಶ * dst_ಖಂಡ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_ಖಂಡ, ದಶಮಾಂಶ scalar, ಪೂರ್ಣಾಂಕ elements) {               \
        ಸ್ಥಿರಾಂಕ int32_t vec_ಕೊನೆ = (elements / 8) * 8;                                                                  \
        ಸ್ಥಿರಾಂಕ int32_t ಬಾಲ    = elements - vec_ಕೊನೆ;                                                                  \
                                                                                                                     \
        ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ;                                                                                     \
        __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));                                                           \
        __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");                                                                    \
                                                                                                                     \
        ಗಾಗಿ (int32_t i = 0; i < vec_ಕೊನೆ; i += 8) {                                                                   \
            __asm__ ಅಸ್ಥಿರ(                                                                                        \
                "flw.ps f10, %[s0]\n"                                                                                \
                "fbc.ps f11, %[s]\n" op_insn                                                                         \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, %[d]\n"                                                                                 \
                : [d] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ಖಂಡ[i])                                                           \
                : [s0] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src0_ಖಂಡ[i]), [s] "m"(scalar)                                   \
                : "f10", "f11", "f12");                                                                              \
        }                                                                                                            \
                                                                                                                     \
        ಆದರೆ (ಬಾಲ > 0) {                                                                                              \
            ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ಬಾಲ_m0 = (1ul << ಬಾಲ) - 1;                                                         \
            __asm__ ಅಸ್ಥಿರ(                                                                                        \
                "mov.m.x m0, %[tm], 0\n"                                                                             \
                "flw.ps f10, 0(%[s0])\n"                                                                             \
                "fbc.ps f11, 0(%[ps])\n" op_insn                                                                     \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, 0(%[d])\n"                                                                              \
                :                                                                                                    \
                : [s0] "r"(&src0_ಖಂಡ[vec_ಕೊನೆ]), [ps] "r"(&scalar), [d] "r"(&dst_ಖಂಡ[vec_ಕೊನೆ]), [tm] "r"(ಬಾಲ_m0) \
                : "f10", "f11", "f12", "memory");                                                                    \
        }                                                                                                            \
                                                                                                                     \
        __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));                                                            \
    }

DEFINE_ಖಂಡ_OP_ಪ್ರಸಾರ(ಖಂಡ_mul_ಪ್ರಸಾರ, "fmul.ps")
DEFINE_ಖಂಡ_OP_ಪ್ರಸಾರ(ಖಂಡ_ಕೂಡಿಸು_ಪ್ರಸಾರ, "fadd.ps")
DEFINE_ಖಂಡ_OP_ಪ್ರಸಾರ(ಖಂಡ_sub_ಪ್ರಸಾರ, "fsub.ps")

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ scalar_el_map(ದಶಮಾಂಶ src0, ದಶಮಾಂಶ src1, ಎಣಿಕೆ ggml_op operation) {
    ಆಯ್ಕೆ (operation) {
        ಸ್ಥಿತಿ GGML_OP_MUL:
            ಹಿಂತಿರುಗಿ src0 * src1;
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
            ಹಿಂತಿರುಗಿ src0 + src1;
        ಸ್ಥಿತಿ GGML_OP_SUB:
            ಹಿಂತಿರುಗಿ src0 - src1;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ 0.0f;
    }
}

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಆದರೆ (!kernel_env) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಪೂರ್ಣಾಂಕ thread_id   = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಎಳೆಗಳು = get_ಅಂಕೆ_ಎಳೆಗಳು(kernel_env->shire_ಮುಖವಾಡ);

    ಆದರೆ (thread_id < 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಆದರೆ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಹಿಂತಿರುಗಿ -1;  // Invalid pointer
    }

    ರಚನೆ ggml_tensor * src0 = &params->src0;
    ರಚನೆ ggml_tensor * src1 = &params->src1;
    ರಚನೆ ggml_tensor * dst  = &params->dst;

    ಆದರೆ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;  // Unsupported type combination
    }

    ದಶಮಾಂಶ * src0_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * src1_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಆದರೆ (!src0_ದತ್ತಾಂಶ || !src1_ದತ್ತಾಂಶ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;  // Null data pointer
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ET_UBERKERNEL
    // Consumer-side input eviction. Required because ET caches are
    // incoherent across minions: if a previous kernel in this UK batch
    // left stale lines for these addresses in this hart's L1, drop them
    // so we read fresh from L3/DRAM (where the producer flushed its
    // results). Standalone launches don't need this -- the host-side
    // runtime boundary between kernel launches handles it.
    ಸ್ಥಿರಾಂಕ size_t src0_ಬೈಟುಗಳು = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    ಸ್ಥಿರಾಂಕ size_t src1_ಬೈಟುಗಳು = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_ದತ್ತಾಂಶ, src0_ಬೈಟುಗಳು);
    evict_region_past_l2(src1_ದತ್ತಾಂಶ, src1_ಬೈಟುಗಳು);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#ಆದರೆ_ಅಂತ್ಯ

    ಎಣಿಕೆ ggml_op operation = dst->op;

    ಆದರೆ (operation != GGML_OP_MUL && operation != GGML_OP_ಕೂಡಿಸು && operation != GGML_OP_SUB) {
        ಹಿಂತಿರುಗಿ -1;  // Unsupported operation
    }

    ಸ್ಥಿರಾಂಕ int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    ಸ್ಥಿರಾಂಕ int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    ಸ್ಥಿರಾಂಕ int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];

    ಸ್ಥಿರಾಂಕ size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ಸ್ಥಿರಾಂಕ size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ಸ್ಥಿರಾಂಕ size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    ಸ್ಥಿರಾಂಕ bool cache_aligned = (dst->ne[0] % 16 == 0);

    // Fast path: no broadcasting, contiguous
    ಸ್ಥಿರಾಂಕ bool no_ಪ್ರಸಾರ = (ne10 == ne0 && ne11 == ne1 && ne12 == ne2 && ne13 == ne3);
    ಸ್ಥಿರಾಂಕ bool all_contiguous =
        (nb0 == 4 && nb00 == 4 && nb10 == 4 && nb1 == ne0 * 4 && nb01 == ne0 * 4 && nb11 == ne0 * 4);

    ಆದರೆ (no_ಪ್ರಸಾರ && all_contiguous) {
        ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_elements         = ne0 * ne1 * ne2 * ne3;
        ಸ್ಥಿರಾಂಕ int64_t elements_ಪ್ರತಿ_cacheline = 16;  // 64 bytes / 4 bytes
        ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_cachelines       = (ಒಟ್ಟು_elements + elements_ಪ್ರತಿ_cacheline - 1) / elements_ಪ್ರತಿ_cacheline;

        ಸ್ಥಿರಾಂಕ int64_t cl_ಪ್ರತಿ_thread = (ಒಟ್ಟು_cachelines + ಅಂಕೆ_ಎಳೆಗಳು - 1) / ಅಂಕೆ_ಎಳೆಗಳು;
        ಸ್ಥಿರಾಂಕ int64_t cl_ಆರಂಭ      = thread_id * cl_ಪ್ರತಿ_thread;
        int64_t       cl_ಕೊನೆ        = cl_ಆರಂಭ + cl_ಪ್ರತಿ_thread;
        ಆದರೆ (cl_ಕೊನೆ > ಒಟ್ಟು_cachelines) {
            cl_ಕೊನೆ = ಒಟ್ಟು_cachelines;
        }

        ಆದರೆ (cl_ಆರಂಭ >= ಒಟ್ಟು_cachelines) {
            ಹಿಂತಿರುಗಿ 0;
        }

        ಸ್ಥಿರಾಂಕ int64_t elem_ಆರಂಭ = cl_ಆರಂಭ * elements_ಪ್ರತಿ_cacheline;
        int64_t       elem_ಕೊನೆ   = cl_ಕೊನೆ * elements_ಪ್ರತಿ_cacheline;
        ಆದರೆ (elem_ಕೊನೆ > ಒಟ್ಟು_elements) {
            elem_ಕೊನೆ = ಒಟ್ಟು_elements;
        }
        ಸ್ಥಿರಾಂಕ int32_t ಎಣಿಕೆ_2 = (int32_t) (elem_ಕೊನೆ - elem_ಆರಂಭ);

        ಆಯ್ಕೆ (operation) {
            ಸ್ಥಿತಿ GGML_OP_MUL:
                ಖಂಡ_mul_cache_aligned(dst_ದತ್ತಾಂಶ + elem_ಆರಂಭ, src0_ದತ್ತಾಂಶ + elem_ಆರಂಭ, src1_ದತ್ತಾಂಶ + elem_ಆರಂಭ, ಎಣಿಕೆ_2);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
                ಖಂಡ_ಕೂಡಿಸು_cache_aligned(dst_ದತ್ತಾಂಶ + elem_ಆರಂಭ, src0_ದತ್ತಾಂಶ + elem_ಆರಂಭ, src1_ದತ್ತಾಂಶ + elem_ಆರಂಭ, ಎಣಿಕೆ_2);
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_SUB:
                ಖಂಡ_sub_cache_aligned(dst_ದತ್ತಾಂಶ + elem_ಆರಂಭ, src0_ದತ್ತಾಂಶ + elem_ಆರಂಭ, src1_ದತ್ತಾಂಶ + elem_ಆರಂಭ, ಎಣಿಕೆ_2);
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                ಹಿಂತಿರುಗಿ 1;
        }
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ET_UBERKERNEL
        // Producer-side flush: ET caches are incoherent across minions, so
        // a consumer kernel running on a different minion can't see our
        // dirty L1 lines via its own evict_region_past_l2. Push our writes
        // all the way to DRAM so the next batched kernel reads fresh.
        // Standalone launches don't need this -- the host runtime boundary
        // between kernel launches handles cache writeback.
        FENCE;
        evict_region_past_l2(dst_ದತ್ತಾಂಶ + elem_ಆರಂಭ, (size_t) ಎಣಿಕೆ_2 * ಗಾತ್ರ(ದಶಮಾಂಶ));
        WAIT_CACHEOPS;
        FENCE;
#ಆದರೆ_ಅಂತ್ಯ
        ಹಿಂತಿರುಗಿ 0;
    }

    // Slow path: broadcasting or non-contiguous
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = ne1 * ne2 * ne3;

    int64_t ಆರಂಭ_ಅಡ್ಡಸಾಲು;
    int64_t ಕೊನೆ_ಅಡ್ಡಸಾಲು;

    ಆದರೆ (cache_aligned) {
        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread = (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು + ಅಂಕೆ_ಎಳೆಗಳು - 1) / ಅಂಕೆ_ಎಳೆಗಳು;
        ಆರಂಭ_ಅಡ್ಡಸಾಲು                     = thread_id * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread;
        ಕೊನೆ_ಅಡ್ಡಸಾಲು = (ಆರಂಭ_ಅಡ್ಡಸಾಲು + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread < ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು) ? (ಆರಂಭ_ಅಡ್ಡಸಾಲು + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_thread) : ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು = et_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_cacheline_ಗುಂಪು(ne0, ಗಾತ್ರ(ದಶಮಾಂಶ));
        ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_groups   = (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು - 1) / ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು;

        ಆದರೆ (thread_id >= ಒಟ್ಟು_groups) {
            ಹಿಂತಿರುಗಿ 0;
        }

        ಸ್ಥಿರಾಂಕ int64_t ಗುಂಪು_ಆರಂಭ = thread_id;
        ಗಾಗಿ (int64_t grp = ಗುಂಪು_ಆರಂಭ; grp < ಒಟ್ಟು_groups; grp += ಅಂಕೆ_ಎಳೆಗಳು) {
            ಸ್ಥಿರಾಂಕ int64_t ಗುಂಪು_ಅಡ್ಡಸಾಲು_ಆರಂಭ = grp * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು;
            int64_t       ಗುಂಪು_ಅಡ್ಡಸಾಲು_ಕೊನೆ   = ಗುಂಪು_ಅಡ್ಡಸಾಲು_ಆರಂಭ + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು;
            ಆದರೆ (ಗುಂಪು_ಅಡ್ಡಸಾಲು_ಕೊನೆ > ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು) {
                ಗುಂಪು_ಅಡ್ಡಸಾಲು_ಕೊನೆ = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
            }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ET_UBERKERNEL
            // First row written by this group (used for producer-side evict).
            ಸ್ಥಿರಾಂಕ int64_t ಮೊದಲ_i03      = ಗುಂಪು_ಅಡ್ಡಸಾಲು_ಆರಂಭ / (ne2 * ne1);
            ಸ್ಥಿರಾಂಕ int64_t ಮೊದಲ_i02      = (ಗುಂಪು_ಅಡ್ಡಸಾಲು_ಆರಂಭ - ಮೊದಲ_i03 * ne2 * ne1) / ne1;
            ಸ್ಥಿರಾಂಕ int64_t ಮೊದಲ_i01      = (ಗುಂಪು_ಅಡ್ಡಸಾಲು_ಆರಂಭ - ಮೊದಲ_i03 * ne2 * ne1 - ಮೊದಲ_i02 * ne1);
            ಅಕ್ಷರ *        ಗುಂಪು_dst_ತಳ = (ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + ಮೊದಲ_i03 * nb3 + ಮೊದಲ_i02 * nb2 + ಮೊದಲ_i01 * nb1;
#ಆದರೆ_ಅಂತ್ಯ

            ಗಾಗಿ (int64_t ir = ಗುಂಪು_ಅಡ್ಡಸಾಲು_ಆರಂಭ; ir < ಗುಂಪು_ಅಡ್ಡಸಾಲು_ಕೊನೆ; ir++) {
                ಸ್ಥಿರಾಂಕ int64_t i03 = ir / (ne2 * ne1);
                ಸ್ಥಿರಾಂಕ int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
                ಸ್ಥಿರಾಂಕ int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

                ಸ್ಥಿರಾಂಕ int64_t i13 = i03 % ne13;
                ಸ್ಥಿರಾಂಕ int64_t i12 = i02 % ne12;
                ಸ್ಥಿರಾಂಕ int64_t i11 = i01 % ne11;

                ದಶಮಾಂಶ *       dst_ptr = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + i03 * nb3 + i02 * nb2 + i01 * nb1);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_ptr =
                    (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i03 * nb03 + i02 * nb02 + i01 * nb01);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src1_ptr =
                    (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + i13 * nb13 + i12 * nb12 + i11 * nb11);

                ಆದರೆ (ne10 == 1) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scalar = src1_ptr[0];
                    ಗಾಗಿ (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], scalar, operation);
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಗಾಗಿ (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], src1_ptr[i0 % ne10], operation);
                    }
                }
            }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ET_UBERKERNEL
            // Producer-side flush for this group's rows. Group rows are
            // contiguous because nb1 = ne0*4 in the cacheline-group layout.
            // Only needed inside a UK batch; see comment in fast path.
            ಸ್ಥಿರಾಂಕ int64_t nrows = ಗುಂಪು_ಅಡ್ಡಸಾಲು_ಕೊನೆ - ಗುಂಪು_ಅಡ್ಡಸಾಲು_ಆರಂಭ;
            ಆದರೆ (nrows > 0) {
                FENCE;
                evict_region_past_l2(ಗುಂಪು_dst_ತಳ, (size_t) nrows * nb1);
                WAIT_CACHEOPS;
                FENCE;
            }
#ಆದರೆ_ಅಂತ್ಯ
        }

        ಹಿಂತಿರುಗಿ 0;
    }

    ಆದರೆ (ಆರಂಭ_ಅಡ್ಡಸಾಲು >= ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಗಾಗಿ (int64_t ir = ಆರಂಭ_ಅಡ್ಡಸಾಲು; ir < ಕೊನೆ_ಅಡ್ಡಸಾಲು; ir++) {
        // Convert flat row index to 3D coordinates
        ಸ್ಥಿರಾಂಕ int64_t i03 = ir / (ne2 * ne1);
        ಸ್ಥಿರಾಂಕ int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
        ಸ್ಥಿರಾಂಕ int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

        // Handle broadcasting: src1 coordinates with modulo
        ಸ್ಥಿರಾಂಕ int64_t i13 = i03 % ne13;
        ಸ್ಥಿರಾಂಕ int64_t i12 = i02 % ne12;
        ಸ್ಥಿರಾಂಕ int64_t i11 = i01 % ne11;

        // Calculate base pointers for this row using stride-based addressing
        ದಶಮಾಂಶ *       dst_ptr  = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + i03 * nb3 + i02 * nb2 + i01 * nb1);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i03 * nb03 + i02 * nb02 + i01 * nb01);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src1_ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + i13 * nb13 + i12 * nb12 + i11 * nb11);

        ಆದರೆ (ne10 == 1) {
            // Broadcast scalar: src1 has ne[0]=1, broadcast across entire row
            ದಶಮಾಂಶ scalar = src1_ptr[0];
            ಆಯ್ಕೆ (operation) {
                ಸ್ಥಿತಿ GGML_OP_MUL:
                    ಖಂಡ_mul_ಪ್ರಸಾರ(dst_ptr, src0_ptr, scalar, (ಪೂರ್ಣಾಂಕ) ne0);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
                    ಖಂಡ_ಕೂಡಿಸು_ಪ್ರಸಾರ(dst_ptr, src0_ptr, scalar, (ಪೂರ್ಣಾಂಕ) ne0);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_OP_SUB:
                    ಖಂಡ_sub_ಪ್ರಸಾರ(dst_ptr, src0_ptr, scalar, (ಪೂರ್ಣಾಂಕ) ne0);
                    ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ 1;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            // Broadcasting in dimension 0: src1 repeats across src0
            ಸ್ಥಿರಾಂಕ int64_t nr0 = ne0 / ne10;

            ಗಾಗಿ (int64_t r = 0; r < nr0; r++) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_ಖಂಡ = src0_ptr + r * ne10;
                ದಶಮಾಂಶ *       dst_ಖಂಡ  = dst_ptr + r * ne10;

                ಆಯ್ಕೆ (operation) {
                    ಸ್ಥಿತಿ GGML_OP_MUL:
                        ಖಂಡ_mul_cache_aligned(dst_ಖಂಡ, src0_ಖಂಡ, src1_ptr, (ಪೂರ್ಣಾಂಕ) ne10);
                        ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
                        ಖಂಡ_ಕೂಡಿಸು_cache_aligned(dst_ಖಂಡ, src0_ಖಂಡ, src1_ptr, (ಪೂರ್ಣಾಂಕ) ne10);
                        ನಿಲ್ಲಿಸು;
                    ಸ್ಥಿತಿ GGML_OP_SUB:
                        ಖಂಡ_sub_cache_aligned(dst_ಖಂಡ, src0_ಖಂಡ, src1_ptr, (ಪೂರ್ಣಾಂಕ) ne10);
                        ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ಹಿಂತಿರುಗಿ 1;
                }
            }
        }
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ET_UBERKERNEL
    // Producer-side flush for the cache-aligned slow path. Rows
    // [start_row, end_row) are contiguous in dst because nb1 = ne0 * 4.
    // Only needed inside a UK batch; see comment in fast path.
    ಆದರೆ (ಕೊನೆ_ಅಡ್ಡಸಾಲು > ಆರಂಭ_ಅಡ್ಡಸಾಲು) {
        FENCE;
        evict_region_past_l2((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + ಆರಂಭ_ಅಡ್ಡಸಾಲು * nb1, (size_t) (ಕೊನೆ_ಅಡ್ಡಸಾಲು - ಆರಂಭ_ಅಡ್ಡಸಾಲು) * nb1);
        WAIT_CACHEOPS;
        FENCE;
    }
#ಆದರೆ_ಅಂತ್ಯ
    ಹಿಂತಿರುಗಿ 0;
}
