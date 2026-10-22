// Element-wise operations: dst[i] = src0[i] op src1[i]
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

// Generic m0-gated element-wise block operation.
// The OP parameter selects the instruction: "fmul.ps", "fadd.ps", "fsub.ps".
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DEFINE_ਖੰਡ_OP(name, op_insn)                                                                             \
    ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ name(ਦਸ਼ਮਲਵ * dst_ਖੰਡ, ਅਚਲ ਦਸ਼ਮਲਵ * src0_ਖੰਡ, ਅਚਲ ਦਸ਼ਮਲਵ * src1_ਖੰਡ, ਪੂਰਨ elements) { \
        ਅਚਲ int32_t vec_ਸਿਰਾ = (elements / 8) * 8;                                                                \
        ਅਚਲ int32_t ਪੂਛ    = elements - vec_ਸਿਰਾ;                                                                \
                                                                                                                   \
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਆਰਜ਼ੀ_ਨਕਾਬ;                                                                                   \
        __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(ਆਰਜ਼ੀ_ਨਕਾਬ));                                                         \
        __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");                                                                  \
                                                                                                                   \
        ਲਈ (int32_t i = 0; i < vec_ਸਿਰਾ; i += 8) {                                                                 \
            __asm__ ਅਸਥਿਰ(                                                                                      \
                "flw.ps f10, %[s0]\n"                                                                              \
                "flw.ps f11, %[s1]\n" op_insn                                                                      \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, %[d]\n"                                                                               \
                : [d] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ਖੰਡ[i])                                                         \
                : [s0] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src0_ਖੰਡ[i]), [s1] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src1_ਖੰਡ[i]) \
                : "f10", "f11", "f12");                                                                            \
        }                                                                                                          \
        /* Deal with tail chunks */                                                                                \
        ਜੇ (ਪੂਛ > 0) {                                                                                            \
            ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਪੂਛ_m0 = (1ul << ਪੂਛ) - 1;                                                       \
            __asm__ ਅਸਥਿਰ(                                                                                      \
                "mov.m.x m0, %[tm], 0\n"                                                                           \
                "flw.ps f10, 0(%[s0])\n"                                                                           \
                "flw.ps f11, 0(%[s1])\n" op_insn                                                                   \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, 0(%[d])\n"                                                                            \
                :                                                                                                  \
                : [s0] "r"(&src0_ਖੰਡ[vec_ਸਿਰਾ]), [s1] "r"(&src1_ਖੰਡ[vec_ਸਿਰਾ]), [d] "r"(&dst_ਖੰਡ[vec_ਸਿਰਾ]),    \
                  [tm] "r"(ਪੂਛ_m0)                                                                                \
                : "f10", "f11", "f12", "memory");                                                                  \
        }                                                                                                          \
                                                                                                                   \
        __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(ਆਰਜ਼ੀ_ਨਕਾਬ));                                                          \
    }

DEFINE_ਖੰਡ_OP(ਖੰਡ_mul_cache_aligned, "fmul.ps")
DEFINE_ਖੰਡ_OP(ਖੰਡ_ਜੋੜੋ_cache_aligned, "fadd.ps")
DEFINE_ਖੰਡ_OP(ਖੰਡ_sub_cache_aligned, "fsub.ps")

// Broadcast variants: src1 is a single scalar, broadcast to all 8 lanes.
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DEFINE_ਖੰਡ_OP_ਪ੍ਰਸਾਰਣ(name, op_insn)                                                                     \
    ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ name(ਦਸ਼ਮਲਵ * dst_ਖੰਡ, ਅਚਲ ਦਸ਼ਮਲਵ * src0_ਖੰਡ, ਦਸ਼ਮਲਵ scalar, ਪੂਰਨ elements) {               \
        ਅਚਲ int32_t vec_ਸਿਰਾ = (elements / 8) * 8;                                                                  \
        ਅਚਲ int32_t ਪੂਛ    = elements - vec_ਸਿਰਾ;                                                                  \
                                                                                                                     \
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਆਰਜ਼ੀ_ਨਕਾਬ;                                                                                     \
        __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(ਆਰਜ਼ੀ_ਨਕਾਬ));                                                           \
        __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");                                                                    \
                                                                                                                     \
        ਲਈ (int32_t i = 0; i < vec_ਸਿਰਾ; i += 8) {                                                                   \
            __asm__ ਅਸਥਿਰ(                                                                                        \
                "flw.ps f10, %[s0]\n"                                                                                \
                "fbc.ps f11, %[s]\n" op_insn                                                                         \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, %[d]\n"                                                                                 \
                : [d] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ਖੰਡ[i])                                                           \
                : [s0] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src0_ਖੰਡ[i]), [s] "m"(scalar)                                   \
                : "f10", "f11", "f12");                                                                              \
        }                                                                                                            \
                                                                                                                     \
        ਜੇ (ਪੂਛ > 0) {                                                                                              \
            ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਪੂਛ_m0 = (1ul << ਪੂਛ) - 1;                                                         \
            __asm__ ਅਸਥਿਰ(                                                                                        \
                "mov.m.x m0, %[tm], 0\n"                                                                             \
                "flw.ps f10, 0(%[s0])\n"                                                                             \
                "fbc.ps f11, 0(%[ps])\n" op_insn                                                                     \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, 0(%[d])\n"                                                                              \
                :                                                                                                    \
                : [s0] "r"(&src0_ਖੰਡ[vec_ਸਿਰਾ]), [ps] "r"(&scalar), [d] "r"(&dst_ਖੰਡ[vec_ਸਿਰਾ]), [tm] "r"(ਪੂਛ_m0) \
                : "f10", "f11", "f12", "memory");                                                                    \
        }                                                                                                            \
                                                                                                                     \
        __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(ਆਰਜ਼ੀ_ਨਕਾਬ));                                                            \
    }

DEFINE_ਖੰਡ_OP_ਪ੍ਰਸਾਰਣ(ਖੰਡ_mul_ਪ੍ਰਸਾਰਣ, "fmul.ps")
DEFINE_ਖੰਡ_OP_ਪ੍ਰਸਾਰਣ(ਖੰਡ_ਜੋੜੋ_ਪ੍ਰਸਾਰਣ, "fadd.ps")
DEFINE_ਖੰਡ_OP_ਪ੍ਰਸਾਰਣ(ਖੰਡ_sub_ਪ੍ਰਸਾਰਣ, "fsub.ps")

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ scalar_el_map(ਦਸ਼ਮਲਵ src0, ਦਸ਼ਮਲਵ src1, ਗਿਣਤੀ ggml_op operation) {
    ਚੋਣ (operation) {
        ਹਾਲਤ GGML_OP_MUL:
            ਵਾਪਸ src0 * src1;
        ਹਾਲਤ GGML_OP_ਜੋੜੋ:
            ਵਾਪਸ src0 + src1;
        ਹਾਲਤ GGML_OP_SUB:
            ਵਾਪਸ src0 - src1;
        ਮੂਲ:
            ਵਾਪਸ 0.0f;
    }
}

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ਜੇ (!kernel_env) {
        ਵਾਪਸ -1;
    }

    ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);

    ਜੇ (thread_id < 0) {
        ਵਾਪਸ 0;
    }

    ਜੇ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;  // Invalid pointer
    }

    ਬਣਤਰ ggml_tensor * src0 = &params->src0;
    ਬਣਤਰ ggml_tensor * src1 = &params->src1;
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;

    ਜੇ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;  // Unsupported type combination
    }

    ਦਸ਼ਮਲਵ * src0_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * src1_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਜੇ (!src0_ਅੰਕੜੇ || !src1_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;  // Null data pointer
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ET_UBERKERNEL
    // Consumer-side input eviction. Required because ET caches are
    // incoherent across minions: if a previous kernel in this UK batch
    // left stale lines for these addresses in this hart's L1, drop them
    // so we read fresh from L3/DRAM (where the producer flushed its
    // results). Standalone launches don't need this -- the host-side
    // runtime boundary between kernel launches handles it.
    ਅਚਲ size_t src0_ਬਾਈਟਾਂ = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    ਅਚਲ size_t src1_ਬਾਈਟਾਂ = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_ਅੰਕੜੇ, src0_ਬਾਈਟਾਂ);
    evict_region_past_l2(src1_ਅੰਕੜੇ, src1_ਬਾਈਟਾਂ);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#ਜੇ_ਅੰਤ

    ਗਿਣਤੀ ggml_op operation = dst->op;

    ਜੇ (operation != GGML_OP_MUL && operation != GGML_OP_ਜੋੜੋ && operation != GGML_OP_SUB) {
        ਵਾਪਸ -1;  // Unsupported operation
    }

    ਅਚਲ int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    ਅਚਲ int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    ਅਚਲ int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];

    ਅਚਲ size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ਅਚਲ size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ਅਚਲ size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    ਅਚਲ bool cache_aligned = (dst->ne[0] % 16 == 0);

    // Fast path: no broadcasting, contiguous
    ਅਚਲ bool no_ਪ੍ਰਸਾਰਣ = (ne10 == ne0 && ne11 == ne1 && ne12 == ne2 && ne13 == ne3);
    ਅਚਲ bool all_contiguous =
        (nb0 == 4 && nb00 == 4 && nb10 == 4 && nb1 == ne0 * 4 && nb01 == ne0 * 4 && nb11 == ne0 * 4);

    ਜੇ (no_ਪ੍ਰਸਾਰਣ && all_contiguous) {
        ਅਚਲ int64_t ਕੁੱਲ_elements         = ne0 * ne1 * ne2 * ne3;
        ਅਚਲ int64_t elements_ਪ੍ਰਤੀ_cacheline = 16;  // 64 bytes / 4 bytes
        ਅਚਲ int64_t ਕੁੱਲ_cachelines       = (ਕੁੱਲ_elements + elements_ਪ੍ਰਤੀ_cacheline - 1) / elements_ਪ੍ਰਤੀ_cacheline;

        ਅਚਲ int64_t cl_ਪ੍ਰਤੀ_thread = (ਕੁੱਲ_cachelines + ਅੰਕ_ਧਾਗੇ - 1) / ਅੰਕ_ਧਾਗੇ;
        ਅਚਲ int64_t cl_ਸ਼ੁਰੂ      = thread_id * cl_ਪ੍ਰਤੀ_thread;
        int64_t       cl_ਸਿਰਾ        = cl_ਸ਼ੁਰੂ + cl_ਪ੍ਰਤੀ_thread;
        ਜੇ (cl_ਸਿਰਾ > ਕੁੱਲ_cachelines) {
            cl_ਸਿਰਾ = ਕੁੱਲ_cachelines;
        }

        ਜੇ (cl_ਸ਼ੁਰੂ >= ਕੁੱਲ_cachelines) {
            ਵਾਪਸ 0;
        }

        ਅਚਲ int64_t elem_ਸ਼ੁਰੂ = cl_ਸ਼ੁਰੂ * elements_ਪ੍ਰਤੀ_cacheline;
        int64_t       elem_ਸਿਰਾ   = cl_ਸਿਰਾ * elements_ਪ੍ਰਤੀ_cacheline;
        ਜੇ (elem_ਸਿਰਾ > ਕੁੱਲ_elements) {
            elem_ਸਿਰਾ = ਕੁੱਲ_elements;
        }
        ਅਚਲ int32_t ਗਿਣਤੀ_2 = (int32_t) (elem_ਸਿਰਾ - elem_ਸ਼ੁਰੂ);

        ਚੋਣ (operation) {
            ਹਾਲਤ GGML_OP_MUL:
                ਖੰਡ_mul_cache_aligned(dst_ਅੰਕੜੇ + elem_ਸ਼ੁਰੂ, src0_ਅੰਕੜੇ + elem_ਸ਼ੁਰੂ, src1_ਅੰਕੜੇ + elem_ਸ਼ੁਰੂ, ਗਿਣਤੀ_2);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_ਜੋੜੋ:
                ਖੰਡ_ਜੋੜੋ_cache_aligned(dst_ਅੰਕੜੇ + elem_ਸ਼ੁਰੂ, src0_ਅੰਕੜੇ + elem_ਸ਼ੁਰੂ, src1_ਅੰਕੜੇ + elem_ਸ਼ੁਰੂ, ਗਿਣਤੀ_2);
                ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_SUB:
                ਖੰਡ_sub_cache_aligned(dst_ਅੰਕੜੇ + elem_ਸ਼ੁਰੂ, src0_ਅੰਕੜੇ + elem_ਸ਼ੁਰੂ, src1_ਅੰਕੜੇ + elem_ਸ਼ੁਰੂ, ਗਿਣਤੀ_2);
                ਰੋਕੋ;
            ਮੂਲ:
                ਵਾਪਸ 1;
        }
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ET_UBERKERNEL
        // Producer-side flush: ET caches are incoherent across minions, so
        // a consumer kernel running on a different minion can't see our
        // dirty L1 lines via its own evict_region_past_l2. Push our writes
        // all the way to DRAM so the next batched kernel reads fresh.
        // Standalone launches don't need this -- the host runtime boundary
        // between kernel launches handles cache writeback.
        FENCE;
        evict_region_past_l2(dst_ਅੰਕੜੇ + elem_ਸ਼ੁਰੂ, (size_t) ਗਿਣਤੀ_2 * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        WAIT_CACHEOPS;
        FENCE;
#ਜੇ_ਅੰਤ
        ਵਾਪਸ 0;
    }

    // Slow path: broadcasting or non-contiguous
    ਅਚਲ int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = ne1 * ne2 * ne3;

    int64_t ਸ਼ੁਰੂ_ਪੰਕਤੀ;
    int64_t ਸਿਰਾ_ਪੰਕਤੀ;

    ਜੇ (cache_aligned) {
        ਅਚਲ int64_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread = (ਕੁੱਲ_ਪੰਕਤੀਆਂ + ਅੰਕ_ਧਾਗੇ - 1) / ਅੰਕ_ਧਾਗੇ;
        ਸ਼ੁਰੂ_ਪੰਕਤੀ                     = thread_id * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread;
        ਸਿਰਾ_ਪੰਕਤੀ = (ਸ਼ੁਰੂ_ਪੰਕਤੀ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread < ਕੁੱਲ_ਪੰਕਤੀਆਂ) ? (ਸ਼ੁਰੂ_ਪੰਕਤੀ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_thread) : ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    } ਨਹੀਂਤਾਂ {
        ਅਚਲ int64_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ = et_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_cacheline_ਸਮੂਹ(ne0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਅਚਲ int64_t ਕੁੱਲ_groups   = (ਕੁੱਲ_ਪੰਕਤੀਆਂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ - 1) / ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ;

        ਜੇ (thread_id >= ਕੁੱਲ_groups) {
            ਵਾਪਸ 0;
        }

        ਅਚਲ int64_t ਸਮੂਹ_ਸ਼ੁਰੂ = thread_id;
        ਲਈ (int64_t grp = ਸਮੂਹ_ਸ਼ੁਰੂ; grp < ਕੁੱਲ_groups; grp += ਅੰਕ_ਧਾਗੇ) {
            ਅਚਲ int64_t ਸਮੂਹ_ਪੰਕਤੀ_ਸ਼ੁਰੂ = grp * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ;
            int64_t       ਸਮੂਹ_ਪੰਕਤੀ_ਸਿਰਾ   = ਸਮੂਹ_ਪੰਕਤੀ_ਸ਼ੁਰੂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਸਮੂਹ;
            ਜੇ (ਸਮੂਹ_ਪੰਕਤੀ_ਸਿਰਾ > ਕੁੱਲ_ਪੰਕਤੀਆਂ) {
                ਸਮੂਹ_ਪੰਕਤੀ_ਸਿਰਾ = ਕੁੱਲ_ਪੰਕਤੀਆਂ;
            }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ET_UBERKERNEL
            // First row written by this group (used for producer-side evict).
            ਅਚਲ int64_t ਪਹਿਲਾ_i03      = ਸਮੂਹ_ਪੰਕਤੀ_ਸ਼ੁਰੂ / (ne2 * ne1);
            ਅਚਲ int64_t ਪਹਿਲਾ_i02      = (ਸਮੂਹ_ਪੰਕਤੀ_ਸ਼ੁਰੂ - ਪਹਿਲਾ_i03 * ne2 * ne1) / ne1;
            ਅਚਲ int64_t ਪਹਿਲਾ_i01      = (ਸਮੂਹ_ਪੰਕਤੀ_ਸ਼ੁਰੂ - ਪਹਿਲਾ_i03 * ne2 * ne1 - ਪਹਿਲਾ_i02 * ne1);
            ਅੱਖਰ *        ਸਮੂਹ_dst_ਅਧਾਰ = (ਅੱਖਰ *) dst_ਅੰਕੜੇ + ਪਹਿਲਾ_i03 * nb3 + ਪਹਿਲਾ_i02 * nb2 + ਪਹਿਲਾ_i01 * nb1;
#ਜੇ_ਅੰਤ

            ਲਈ (int64_t ir = ਸਮੂਹ_ਪੰਕਤੀ_ਸ਼ੁਰੂ; ir < ਸਮੂਹ_ਪੰਕਤੀ_ਸਿਰਾ; ir++) {
                ਅਚਲ int64_t i03 = ir / (ne2 * ne1);
                ਅਚਲ int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
                ਅਚਲ int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

                ਅਚਲ int64_t i13 = i03 % ne13;
                ਅਚਲ int64_t i12 = i02 % ne12;
                ਅਚਲ int64_t i11 = i01 % ne11;

                ਦਸ਼ਮਲਵ *       dst_ptr = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + i03 * nb3 + i02 * nb2 + i01 * nb1);
                ਅਚਲ ਦਸ਼ਮਲਵ * src0_ptr =
                    (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i03 * nb03 + i02 * nb02 + i01 * nb01);
                ਅਚਲ ਦਸ਼ਮਲਵ * src1_ptr =
                    (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + i13 * nb13 + i12 * nb12 + i11 * nb11);

                ਜੇ (ne10 == 1) {
                    ਅਚਲ ਦਸ਼ਮਲਵ scalar = src1_ptr[0];
                    ਲਈ (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], scalar, operation);
                    }
                } ਨਹੀਂਤਾਂ {
                    ਲਈ (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], src1_ptr[i0 % ne10], operation);
                    }
                }
            }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ET_UBERKERNEL
            // Producer-side flush for this group's rows. Group rows are
            // contiguous because nb1 = ne0*4 in the cacheline-group layout.
            // Only needed inside a UK batch; see comment in fast path.
            ਅਚਲ int64_t nrows = ਸਮੂਹ_ਪੰਕਤੀ_ਸਿਰਾ - ਸਮੂਹ_ਪੰਕਤੀ_ਸ਼ੁਰੂ;
            ਜੇ (nrows > 0) {
                FENCE;
                evict_region_past_l2(ਸਮੂਹ_dst_ਅਧਾਰ, (size_t) nrows * nb1);
                WAIT_CACHEOPS;
                FENCE;
            }
#ਜੇ_ਅੰਤ
        }

        ਵਾਪਸ 0;
    }

    ਜੇ (ਸ਼ੁਰੂ_ਪੰਕਤੀ >= ਕੁੱਲ_ਪੰਕਤੀਆਂ) {
        ਵਾਪਸ 0;
    }

    ਲਈ (int64_t ir = ਸ਼ੁਰੂ_ਪੰਕਤੀ; ir < ਸਿਰਾ_ਪੰਕਤੀ; ir++) {
        // Convert flat row index to 3D coordinates
        ਅਚਲ int64_t i03 = ir / (ne2 * ne1);
        ਅਚਲ int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
        ਅਚਲ int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

        // Handle broadcasting: src1 coordinates with modulo
        ਅਚਲ int64_t i13 = i03 % ne13;
        ਅਚਲ int64_t i12 = i02 % ne12;
        ਅਚਲ int64_t i11 = i01 % ne11;

        // Calculate base pointers for this row using stride-based addressing
        ਦਸ਼ਮਲਵ *       dst_ptr  = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + i03 * nb3 + i02 * nb2 + i01 * nb1);
        ਅਚਲ ਦਸ਼ਮਲਵ * src0_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i03 * nb03 + i02 * nb02 + i01 * nb01);
        ਅਚਲ ਦਸ਼ਮਲਵ * src1_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src1_ਅੰਕੜੇ + i13 * nb13 + i12 * nb12 + i11 * nb11);

        ਜੇ (ne10 == 1) {
            // Broadcast scalar: src1 has ne[0]=1, broadcast across entire row
            ਦਸ਼ਮਲਵ scalar = src1_ptr[0];
            ਚੋਣ (operation) {
                ਹਾਲਤ GGML_OP_MUL:
                    ਖੰਡ_mul_ਪ੍ਰਸਾਰਣ(dst_ptr, src0_ptr, scalar, (ਪੂਰਨ) ne0);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_ਜੋੜੋ:
                    ਖੰਡ_ਜੋੜੋ_ਪ੍ਰਸਾਰਣ(dst_ptr, src0_ptr, scalar, (ਪੂਰਨ) ne0);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_OP_SUB:
                    ਖੰਡ_sub_ਪ੍ਰਸਾਰਣ(dst_ptr, src0_ptr, scalar, (ਪੂਰਨ) ne0);
                    ਰੋਕੋ;
                ਮੂਲ:
                    ਵਾਪਸ 1;
            }
        } ਨਹੀਂਤਾਂ {
            // Broadcasting in dimension 0: src1 repeats across src0
            ਅਚਲ int64_t nr0 = ne0 / ne10;

            ਲਈ (int64_t r = 0; r < nr0; r++) {
                ਅਚਲ ਦਸ਼ਮਲਵ * src0_ਖੰਡ = src0_ptr + r * ne10;
                ਦਸ਼ਮਲਵ *       dst_ਖੰਡ  = dst_ptr + r * ne10;

                ਚੋਣ (operation) {
                    ਹਾਲਤ GGML_OP_MUL:
                        ਖੰਡ_mul_cache_aligned(dst_ਖੰਡ, src0_ਖੰਡ, src1_ptr, (ਪੂਰਨ) ne10);
                        ਰੋਕੋ;
                    ਹਾਲਤ GGML_OP_ਜੋੜੋ:
                        ਖੰਡ_ਜੋੜੋ_cache_aligned(dst_ਖੰਡ, src0_ਖੰਡ, src1_ptr, (ਪੂਰਨ) ne10);
                        ਰੋਕੋ;
                    ਹਾਲਤ GGML_OP_SUB:
                        ਖੰਡ_sub_cache_aligned(dst_ਖੰਡ, src0_ਖੰਡ, src1_ptr, (ਪੂਰਨ) ne10);
                        ਰੋਕੋ;
                    ਮੂਲ:
                        ਵਾਪਸ 1;
                }
            }
        }
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ET_UBERKERNEL
    // Producer-side flush for the cache-aligned slow path. Rows
    // [start_row, end_row) are contiguous in dst because nb1 = ne0 * 4.
    // Only needed inside a UK batch; see comment in fast path.
    ਜੇ (ਸਿਰਾ_ਪੰਕਤੀ > ਸ਼ੁਰੂ_ਪੰਕਤੀ) {
        FENCE;
        evict_region_past_l2((ਅੱਖਰ *) dst_ਅੰਕੜੇ + ਸ਼ੁਰੂ_ਪੰਕਤੀ * nb1, (size_t) (ਸਿਰਾ_ਪੰਕਤੀ - ਸ਼ੁਰੂ_ਪੰਕਤੀ) * nb1);
        WAIT_CACHEOPS;
        FENCE;
    }
#ਜੇ_ਅੰਤ
    ਵਾਪਸ 0;
}
