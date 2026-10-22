// Element-wise operations: dst[i] = src0[i] op src1[i]
#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>

// Generic m0-gated element-wise block operation.
// The OP parameter selects the instruction: "fmul.ps", "fadd.ps", "fsub.ps".
#నిర్వచించు DEFINE_ఖండం_OP(name, op_insn)                                                                             \
    స్థిర అంతఃవరుస శూన్య name(దశాంశం * dst_ఖండం, స్థిరరాశి దశాంశం * src0_ఖండం, స్థిరరాశి దశాంశం * src1_ఖండం, పూర్ణాంకం elements) { \
        స్థిరరాశి int32_t vec_చివర = (elements / 8) * 8;                                                                \
        స్థిరరాశి int32_t తోక    = elements - vec_చివర;                                                                \
                                                                                                                   \
        గుర్తులేని దీర్ఘ తాత్కాలిక_ముసుగు;                                                                                   \
        __asm__ అస్థిర("mova.x.m %0" : "=r"(తాత్కాలిక_ముసుగు));                                                         \
        __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");                                                                  \
                                                                                                                   \
        కొరకు (int32_t i = 0; i < vec_చివర; i += 8) {                                                                 \
            __asm__ అస్థిర(                                                                                      \
                "flw.ps f10, %[s0]\n"                                                                              \
                "flw.ps f11, %[s1]\n" op_insn                                                                      \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, %[d]\n"                                                                               \
                : [d] "=m"(*(దశాంశం (*)[8]) & dst_ఖండం[i])                                                         \
                : [s0] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & src0_ఖండం[i]), [s1] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & src1_ఖండం[i]) \
                : "f10", "f11", "f12");                                                                            \
        }                                                                                                          \
        /* Deal with tail chunks */                                                                                \
        ఒకవేళ (తోక > 0) {                                                                                            \
            స్థిరరాశి గుర్తులేని దీర్ఘ తోక_m0 = (1ul << తోక) - 1;                                                       \
            __asm__ అస్థిర(                                                                                      \
                "mov.m.x m0, %[tm], 0\n"                                                                           \
                "flw.ps f10, 0(%[s0])\n"                                                                           \
                "flw.ps f11, 0(%[s1])\n" op_insn                                                                   \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, 0(%[d])\n"                                                                            \
                :                                                                                                  \
                : [s0] "r"(&src0_ఖండం[vec_చివర]), [s1] "r"(&src1_ఖండం[vec_చివర]), [d] "r"(&dst_ఖండం[vec_చివర]),    \
                  [tm] "r"(తోక_m0)                                                                                \
                : "f10", "f11", "f12", "memory");                                                                  \
        }                                                                                                          \
                                                                                                                   \
        __asm__ అస్థిర("mova.m.x %0" ::"r"(తాత్కాలిక_ముసుగు));                                                          \
    }

DEFINE_ఖండం_OP(ఖండం_mul_cache_aligned, "fmul.ps")
DEFINE_ఖండం_OP(ఖండం_కూడు_cache_aligned, "fadd.ps")
DEFINE_ఖండం_OP(ఖండం_sub_cache_aligned, "fsub.ps")

// Broadcast variants: src1 is a single scalar, broadcast to all 8 lanes.
#నిర్వచించు DEFINE_ఖండం_OP_ప్రసారం(name, op_insn)                                                                     \
    స్థిర అంతఃవరుస శూన్య name(దశాంశం * dst_ఖండం, స్థిరరాశి దశాంశం * src0_ఖండం, దశాంశం scalar, పూర్ణాంకం elements) {               \
        స్థిరరాశి int32_t vec_చివర = (elements / 8) * 8;                                                                  \
        స్థిరరాశి int32_t తోక    = elements - vec_చివర;                                                                  \
                                                                                                                     \
        గుర్తులేని దీర్ఘ తాత్కాలిక_ముసుగు;                                                                                     \
        __asm__ అస్థిర("mova.x.m %0" : "=r"(తాత్కాలిక_ముసుగు));                                                           \
        __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");                                                                    \
                                                                                                                     \
        కొరకు (int32_t i = 0; i < vec_చివర; i += 8) {                                                                   \
            __asm__ అస్థిర(                                                                                        \
                "flw.ps f10, %[s0]\n"                                                                                \
                "fbc.ps f11, %[s]\n" op_insn                                                                         \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, %[d]\n"                                                                                 \
                : [d] "=m"(*(దశాంశం (*)[8]) & dst_ఖండం[i])                                                           \
                : [s0] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & src0_ఖండం[i]), [s] "m"(scalar)                                   \
                : "f10", "f11", "f12");                                                                              \
        }                                                                                                            \
                                                                                                                     \
        ఒకవేళ (తోక > 0) {                                                                                              \
            స్థిరరాశి గుర్తులేని దీర్ఘ తోక_m0 = (1ul << తోక) - 1;                                                         \
            __asm__ అస్థిర(                                                                                        \
                "mov.m.x m0, %[tm], 0\n"                                                                             \
                "flw.ps f10, 0(%[s0])\n"                                                                             \
                "fbc.ps f11, 0(%[ps])\n" op_insn                                                                     \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, 0(%[d])\n"                                                                              \
                :                                                                                                    \
                : [s0] "r"(&src0_ఖండం[vec_చివర]), [ps] "r"(&scalar), [d] "r"(&dst_ఖండం[vec_చివర]), [tm] "r"(తోక_m0) \
                : "f10", "f11", "f12", "memory");                                                                    \
        }                                                                                                            \
                                                                                                                     \
        __asm__ అస్థిర("mova.m.x %0" ::"r"(తాత్కాలిక_ముసుగు));                                                            \
    }

DEFINE_ఖండం_OP_ప్రసారం(ఖండం_mul_ప్రసారం, "fmul.ps")
DEFINE_ఖండం_OP_ప్రసారం(ఖండం_కూడు_ప్రసారం, "fadd.ps")
DEFINE_ఖండం_OP_ప్రసారం(ఖండం_sub_ప్రసారం, "fsub.ps")

స్థిర అంతఃవరుస దశాంశం scalar_el_map(దశాంశం src0, దశాంశం src1, గణన ggml_op operation) {
    ఎంపిక (operation) {
        స్థితి GGML_OP_MUL:
            తిరిగి src0 * src1;
        స్థితి GGML_OP_కూడు:
            తిరిగి src0 + src1;
        స్థితి GGML_OP_SUB:
            తిరిగి src0 - src1;
        అప్రమేయ:
            తిరిగి 0.0f;
    }
}

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_ద్విమాన_params * params, శూన్య * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ఒకవేళ (!kernel_env) {
        తిరిగి -1;
    }

    పూర్ణాంకం thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    పూర్ణాంకం అంకె_దారాలు = get_అంకె_దారాలు(kernel_env->shire_ముసుగు);

    ఒకవేళ (thread_id < 0) {
        తిరిగి 0;
    }

    ఒకవేళ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        తిరిగి -1;  // Invalid pointer
    }

    నిర్మాణం ggml_tensor * src0 = &params->src0;
    నిర్మాణం ggml_tensor * src1 = &params->src1;
    నిర్మాణం ggml_tensor * dst  = &params->dst;

    ఒకవేళ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి -1;  // Unsupported type combination
    }

    దశాంశం * src0_దత్తాంశం = (దశాంశం *) src0->దత్తాంశం;
    దశాంశం * src1_దత్తాంశం = (దశాంశం *) src1->దత్తాంశం;
    దశాంశం * dst_దత్తాంశం  = (దశాంశం *) dst->దత్తాంశం;

    ఒకవేళ (!src0_దత్తాంశం || !src1_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;  // Null data pointer
    }

#నిర్వచితమైతే ET_UBERKERNEL
    // Consumer-side input eviction. Required because ET caches are
    // incoherent across minions: if a previous kernel in this UK batch
    // left stale lines for these addresses in this hart's L1, drop them
    // so we read fresh from L3/DRAM (where the producer flushed its
    // results). Standalone launches don't need this -- the host-side
    // runtime boundary between kernel launches handles it.
    స్థిరరాశి size_t src0_బైట్లు = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    స్థిరరాశి size_t src1_బైట్లు = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_దత్తాంశం, src0_బైట్లు);
    evict_region_past_l2(src1_దత్తాంశం, src1_బైట్లు);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#అయితే_ముగింపు

    గణన ggml_op operation = dst->op;

    ఒకవేళ (operation != GGML_OP_MUL && operation != GGML_OP_కూడు && operation != GGML_OP_SUB) {
        తిరిగి -1;  // Unsupported operation
    }

    స్థిరరాశి int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    స్థిరరాశి int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    స్థిరరాశి int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];

    స్థిరరాశి size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    స్థిరరాశి size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    స్థిరరాశి size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    స్థిరరాశి bool cache_aligned = (dst->ne[0] % 16 == 0);

    // Fast path: no broadcasting, contiguous
    స్థిరరాశి bool no_ప్రసారం = (ne10 == ne0 && ne11 == ne1 && ne12 == ne2 && ne13 == ne3);
    స్థిరరాశి bool all_contiguous =
        (nb0 == 4 && nb00 == 4 && nb10 == 4 && nb1 == ne0 * 4 && nb01 == ne0 * 4 && nb11 == ne0 * 4);

    ఒకవేళ (no_ప్రసారం && all_contiguous) {
        స్థిరరాశి int64_t మొత్తం_elements         = ne0 * ne1 * ne2 * ne3;
        స్థిరరాశి int64_t elements_చొప్పున_cacheline = 16;  // 64 bytes / 4 bytes
        స్థిరరాశి int64_t మొత్తం_cachelines       = (మొత్తం_elements + elements_చొప్పున_cacheline - 1) / elements_చొప్పున_cacheline;

        స్థిరరాశి int64_t cl_చొప్పున_thread = (మొత్తం_cachelines + అంకె_దారాలు - 1) / అంకె_దారాలు;
        స్థిరరాశి int64_t cl_ప్రారంభం      = thread_id * cl_చొప్పున_thread;
        int64_t       cl_చివర        = cl_ప్రారంభం + cl_చొప్పున_thread;
        ఒకవేళ (cl_చివర > మొత్తం_cachelines) {
            cl_చివర = మొత్తం_cachelines;
        }

        ఒకవేళ (cl_ప్రారంభం >= మొత్తం_cachelines) {
            తిరిగి 0;
        }

        స్థిరరాశి int64_t elem_ప్రారంభం = cl_ప్రారంభం * elements_చొప్పున_cacheline;
        int64_t       elem_చివర   = cl_చివర * elements_చొప్పున_cacheline;
        ఒకవేళ (elem_చివర > మొత్తం_elements) {
            elem_చివర = మొత్తం_elements;
        }
        స్థిరరాశి int32_t లెక్క = (int32_t) (elem_చివర - elem_ప్రారంభం);

        ఎంపిక (operation) {
            స్థితి GGML_OP_MUL:
                ఖండం_mul_cache_aligned(dst_దత్తాంశం + elem_ప్రారంభం, src0_దత్తాంశం + elem_ప్రారంభం, src1_దత్తాంశం + elem_ప్రారంభం, లెక్క);
                ఆపు;
            స్థితి GGML_OP_కూడు:
                ఖండం_కూడు_cache_aligned(dst_దత్తాంశం + elem_ప్రారంభం, src0_దత్తాంశం + elem_ప్రారంభం, src1_దత్తాంశం + elem_ప్రారంభం, లెక్క);
                ఆపు;
            స్థితి GGML_OP_SUB:
                ఖండం_sub_cache_aligned(dst_దత్తాంశం + elem_ప్రారంభం, src0_దత్తాంశం + elem_ప్రారంభం, src1_దత్తాంశం + elem_ప్రారంభం, లెక్క);
                ఆపు;
            అప్రమేయ:
                తిరిగి 1;
        }
#నిర్వచితమైతే ET_UBERKERNEL
        // Producer-side flush: ET caches are incoherent across minions, so
        // a consumer kernel running on a different minion can't see our
        // dirty L1 lines via its own evict_region_past_l2. Push our writes
        // all the way to DRAM so the next batched kernel reads fresh.
        // Standalone launches don't need this -- the host runtime boundary
        // between kernel launches handles cache writeback.
        FENCE;
        evict_region_past_l2(dst_దత్తాంశం + elem_ప్రారంభం, (size_t) లెక్క * పరిమాణం(దశాంశం));
        WAIT_CACHEOPS;
        FENCE;
#అయితే_ముగింపు
        తిరిగి 0;
    }

    // Slow path: broadcasting or non-contiguous
    స్థిరరాశి int64_t మొత్తం_అడ్డువరుసలు = ne1 * ne2 * ne3;

    int64_t ప్రారంభం_అడ్డువరుస;
    int64_t చివర_అడ్డువరుస;

    ఒకవేళ (cache_aligned) {
        స్థిరరాశి int64_t అడ్డువరుసలు_చొప్పున_thread = (మొత్తం_అడ్డువరుసలు + అంకె_దారాలు - 1) / అంకె_దారాలు;
        ప్రారంభం_అడ్డువరుస                     = thread_id * అడ్డువరుసలు_చొప్పున_thread;
        చివర_అడ్డువరుస = (ప్రారంభం_అడ్డువరుస + అడ్డువరుసలు_చొప్పున_thread < మొత్తం_అడ్డువరుసలు) ? (ప్రారంభం_అడ్డువరుస + అడ్డువరుసలు_చొప్పున_thread) : మొత్తం_అడ్డువరుసలు;
    } లేకపోతే {
        స్థిరరాశి int64_t అడ్డువరుసలు_చొప్పున_సమూహం = et_అడ్డువరుసలు_చొప్పున_cacheline_సమూహం(ne0, పరిమాణం(దశాంశం));
        స్థిరరాశి int64_t మొత్తం_groups   = (మొత్తం_అడ్డువరుసలు + అడ్డువరుసలు_చొప్పున_సమూహం - 1) / అడ్డువరుసలు_చొప్పున_సమూహం;

        ఒకవేళ (thread_id >= మొత్తం_groups) {
            తిరిగి 0;
        }

        స్థిరరాశి int64_t సమూహం_ప్రారంభం = thread_id;
        కొరకు (int64_t grp = సమూహం_ప్రారంభం; grp < మొత్తం_groups; grp += అంకె_దారాలు) {
            స్థిరరాశి int64_t సమూహం_అడ్డువరుస_ప్రారంభం = grp * అడ్డువరుసలు_చొప్పున_సమూహం;
            int64_t       సమూహం_అడ్డువరుస_చివర   = సమూహం_అడ్డువరుస_ప్రారంభం + అడ్డువరుసలు_చొప్పున_సమూహం;
            ఒకవేళ (సమూహం_అడ్డువరుస_చివర > మొత్తం_అడ్డువరుసలు) {
                సమూహం_అడ్డువరుస_చివర = మొత్తం_అడ్డువరుసలు;
            }

#నిర్వచితమైతే ET_UBERKERNEL
            // First row written by this group (used for producer-side evict).
            స్థిరరాశి int64_t మొదటి_i03      = సమూహం_అడ్డువరుస_ప్రారంభం / (ne2 * ne1);
            స్థిరరాశి int64_t మొదటి_i02      = (సమూహం_అడ్డువరుస_ప్రారంభం - మొదటి_i03 * ne2 * ne1) / ne1;
            స్థిరరాశి int64_t మొదటి_i01      = (సమూహం_అడ్డువరుస_ప్రారంభం - మొదటి_i03 * ne2 * ne1 - మొదటి_i02 * ne1);
            అక్షరం *        సమూహం_dst_పునాది = (అక్షరం *) dst_దత్తాంశం + మొదటి_i03 * nb3 + మొదటి_i02 * nb2 + మొదటి_i01 * nb1;
#అయితే_ముగింపు

            కొరకు (int64_t ir = సమూహం_అడ్డువరుస_ప్రారంభం; ir < సమూహం_అడ్డువరుస_చివర; ir++) {
                స్థిరరాశి int64_t i03 = ir / (ne2 * ne1);
                స్థిరరాశి int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
                స్థిరరాశి int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

                స్థిరరాశి int64_t i13 = i03 % ne13;
                స్థిరరాశి int64_t i12 = i02 % ne12;
                స్థిరరాశి int64_t i11 = i01 % ne11;

                దశాంశం *       dst_ptr = (దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + i03 * nb3 + i02 * nb2 + i01 * nb1);
                స్థిరరాశి దశాంశం * src0_ptr =
                    (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i03 * nb03 + i02 * nb02 + i01 * nb01);
                స్థిరరాశి దశాంశం * src1_ptr =
                    (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1_దత్తాంశం + i13 * nb13 + i12 * nb12 + i11 * nb11);

                ఒకవేళ (ne10 == 1) {
                    స్థిరరాశి దశాంశం scalar = src1_ptr[0];
                    కొరకు (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], scalar, operation);
                    }
                } లేకపోతే {
                    కొరకు (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], src1_ptr[i0 % ne10], operation);
                    }
                }
            }

#నిర్వచితమైతే ET_UBERKERNEL
            // Producer-side flush for this group's rows. Group rows are
            // contiguous because nb1 = ne0*4 in the cacheline-group layout.
            // Only needed inside a UK batch; see comment in fast path.
            స్థిరరాశి int64_t nrows = సమూహం_అడ్డువరుస_చివర - సమూహం_అడ్డువరుస_ప్రారంభం;
            ఒకవేళ (nrows > 0) {
                FENCE;
                evict_region_past_l2(సమూహం_dst_పునాది, (size_t) nrows * nb1);
                WAIT_CACHEOPS;
                FENCE;
            }
#అయితే_ముగింపు
        }

        తిరిగి 0;
    }

    ఒకవేళ (ప్రారంభం_అడ్డువరుస >= మొత్తం_అడ్డువరుసలు) {
        తిరిగి 0;
    }

    కొరకు (int64_t ir = ప్రారంభం_అడ్డువరుస; ir < చివర_అడ్డువరుస; ir++) {
        // Convert flat row index to 3D coordinates
        స్థిరరాశి int64_t i03 = ir / (ne2 * ne1);
        స్థిరరాశి int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
        స్థిరరాశి int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

        // Handle broadcasting: src1 coordinates with modulo
        స్థిరరాశి int64_t i13 = i03 % ne13;
        స్థిరరాశి int64_t i12 = i02 % ne12;
        స్థిరరాశి int64_t i11 = i01 % ne11;

        // Calculate base pointers for this row using stride-based addressing
        దశాంశం *       dst_ptr  = (దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + i03 * nb3 + i02 * nb2 + i01 * nb1);
        స్థిరరాశి దశాంశం * src0_ptr = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i03 * nb03 + i02 * nb02 + i01 * nb01);
        స్థిరరాశి దశాంశం * src1_ptr = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1_దత్తాంశం + i13 * nb13 + i12 * nb12 + i11 * nb11);

        ఒకవేళ (ne10 == 1) {
            // Broadcast scalar: src1 has ne[0]=1, broadcast across entire row
            దశాంశం scalar = src1_ptr[0];
            ఎంపిక (operation) {
                స్థితి GGML_OP_MUL:
                    ఖండం_mul_ప్రసారం(dst_ptr, src0_ptr, scalar, (పూర్ణాంకం) ne0);
                    ఆపు;
                స్థితి GGML_OP_కూడు:
                    ఖండం_కూడు_ప్రసారం(dst_ptr, src0_ptr, scalar, (పూర్ణాంకం) ne0);
                    ఆపు;
                స్థితి GGML_OP_SUB:
                    ఖండం_sub_ప్రసారం(dst_ptr, src0_ptr, scalar, (పూర్ణాంకం) ne0);
                    ఆపు;
                అప్రమేయ:
                    తిరిగి 1;
            }
        } లేకపోతే {
            // Broadcasting in dimension 0: src1 repeats across src0
            స్థిరరాశి int64_t nr0 = ne0 / ne10;

            కొరకు (int64_t r = 0; r < nr0; r++) {
                స్థిరరాశి దశాంశం * src0_ఖండం = src0_ptr + r * ne10;
                దశాంశం *       dst_ఖండం  = dst_ptr + r * ne10;

                ఎంపిక (operation) {
                    స్థితి GGML_OP_MUL:
                        ఖండం_mul_cache_aligned(dst_ఖండం, src0_ఖండం, src1_ptr, (పూర్ణాంకం) ne10);
                        ఆపు;
                    స్థితి GGML_OP_కూడు:
                        ఖండం_కూడు_cache_aligned(dst_ఖండం, src0_ఖండం, src1_ptr, (పూర్ణాంకం) ne10);
                        ఆపు;
                    స్థితి GGML_OP_SUB:
                        ఖండం_sub_cache_aligned(dst_ఖండం, src0_ఖండం, src1_ptr, (పూర్ణాంకం) ne10);
                        ఆపు;
                    అప్రమేయ:
                        తిరిగి 1;
                }
            }
        }
    }

#నిర్వచితమైతే ET_UBERKERNEL
    // Producer-side flush for the cache-aligned slow path. Rows
    // [start_row, end_row) are contiguous in dst because nb1 = ne0 * 4.
    // Only needed inside a UK batch; see comment in fast path.
    ఒకవేళ (చివర_అడ్డువరుస > ప్రారంభం_అడ్డువరుస) {
        FENCE;
        evict_region_past_l2((అక్షరం *) dst_దత్తాంశం + ప్రారంభం_అడ్డువరుస * nb1, (size_t) (చివర_అడ్డువరుస - ప్రారంభం_అడ్డువరుస) * nb1);
        WAIT_CACHEOPS;
        FENCE;
    }
#అయితే_ముగింపు
    తిరిగి 0;
}
