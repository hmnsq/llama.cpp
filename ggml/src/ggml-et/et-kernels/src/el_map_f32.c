// Element-wise operations: dst[i] = src0[i] op src1[i]
#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

// Generic m0-gated element-wise block operation.
// The OP parameter selects the instruction: "fmul.ps", "fadd.ps", "fsub.ps".
#परिभाषय DEFINE_खण्डम्_OP(name, op_insn)                                                                             \
    स्थिर अन्तःस्थ शून्य name(भिन्न * dst_खण्डम्, ध्रुव भिन्न * src0_खण्डम्, ध्रुव भिन्न * src1_खण्डम्, पूर्णाङ्क elements) { \
        ध्रुव int32_t vec_अन्तः = (elements / 8) * 8;                                                                \
        ध्रुव int32_t पुच्छ    = elements - vec_अन्तः;                                                                \
                                                                                                                   \
        अचिह्नित दीर्घ अस्थायी_आवरणम्;                                                                                   \
        __asm__ चञ्चल("mova.x.m %0" : "=r"(अस्थायी_आवरणम्));                                                         \
        __asm__ चञ्चल("mov.m.x m0, x0, 0xFF");                                                                  \
                                                                                                                   \
        प्रति (int32_t i = 0; i < vec_अन्तः; i += 8) {                                                                 \
            __asm__ चञ्चल(                                                                                      \
                "flw.ps f10, %[s0]\n"                                                                              \
                "flw.ps f11, %[s1]\n" op_insn                                                                      \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, %[d]\n"                                                                               \
                : [d] "=m"(*(भिन्न (*)[8]) & dst_खण्डम्[i])                                                         \
                : [s0] "m"(*(ध्रुव भिन्न (*)[8]) & src0_खण्डम्[i]), [s1] "m"(*(ध्रुव भिन्न (*)[8]) & src1_खण्डम्[i]) \
                : "f10", "f11", "f12");                                                                            \
        }                                                                                                          \
        /* Deal with tail chunks */                                                                                \
        यदि (पुच्छ > 0) {                                                                                            \
            ध्रुव अचिह्नित दीर्घ पुच्छ_m0 = (1ul << पुच्छ) - 1;                                                       \
            __asm__ चञ्चल(                                                                                      \
                "mov.m.x m0, %[tm], 0\n"                                                                           \
                "flw.ps f10, 0(%[s0])\n"                                                                           \
                "flw.ps f11, 0(%[s1])\n" op_insn                                                                   \
                " f12, f10, f11\n"                                                                                 \
                "fsw.ps f12, 0(%[d])\n"                                                                            \
                :                                                                                                  \
                : [s0] "r"(&src0_खण्डम्[vec_अन्तः]), [s1] "r"(&src1_खण्डम्[vec_अन्तः]), [d] "r"(&dst_खण्डम्[vec_अन्तः]),    \
                  [tm] "r"(पुच्छ_m0)                                                                                \
                : "f10", "f11", "f12", "memory");                                                                  \
        }                                                                                                          \
                                                                                                                   \
        __asm__ चञ्चल("mova.m.x %0" ::"r"(अस्थायी_आवरणम्));                                                          \
    }

DEFINE_खण्डम्_OP(खण्डम्_mul_cache_aligned, "fmul.ps")
DEFINE_खण्डम्_OP(खण्डम्_सङ्कलय_cache_aligned, "fadd.ps")
DEFINE_खण्डम्_OP(खण्डम्_sub_cache_aligned, "fsub.ps")

// Broadcast variants: src1 is a single scalar, broadcast to all 8 lanes.
#परिभाषय DEFINE_खण्डम्_OP_प्रसारणम्(name, op_insn)                                                                     \
    स्थिर अन्तःस्थ शून्य name(भिन्न * dst_खण्डम्, ध्रुव भिन्न * src0_खण्डम्, भिन्न scalar, पूर्णाङ्क elements) {               \
        ध्रुव int32_t vec_अन्तः = (elements / 8) * 8;                                                                  \
        ध्रुव int32_t पुच्छ    = elements - vec_अन्तः;                                                                  \
                                                                                                                     \
        अचिह्नित दीर्घ अस्थायी_आवरणम्;                                                                                     \
        __asm__ चञ्चल("mova.x.m %0" : "=r"(अस्थायी_आवरणम्));                                                           \
        __asm__ चञ्चल("mov.m.x m0, x0, 0xFF");                                                                    \
                                                                                                                     \
        प्रति (int32_t i = 0; i < vec_अन्तः; i += 8) {                                                                   \
            __asm__ चञ्चल(                                                                                        \
                "flw.ps f10, %[s0]\n"                                                                                \
                "fbc.ps f11, %[s]\n" op_insn                                                                         \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, %[d]\n"                                                                                 \
                : [d] "=m"(*(भिन्न (*)[8]) & dst_खण्डम्[i])                                                           \
                : [s0] "m"(*(ध्रुव भिन्न (*)[8]) & src0_खण्डम्[i]), [s] "m"(scalar)                                   \
                : "f10", "f11", "f12");                                                                              \
        }                                                                                                            \
                                                                                                                     \
        यदि (पुच्छ > 0) {                                                                                              \
            ध्रुव अचिह्नित दीर्घ पुच्छ_m0 = (1ul << पुच्छ) - 1;                                                         \
            __asm__ चञ्चल(                                                                                        \
                "mov.m.x m0, %[tm], 0\n"                                                                             \
                "flw.ps f10, 0(%[s0])\n"                                                                             \
                "fbc.ps f11, 0(%[ps])\n" op_insn                                                                     \
                " f12, f10, f11\n"                                                                                   \
                "fsw.ps f12, 0(%[d])\n"                                                                              \
                :                                                                                                    \
                : [s0] "r"(&src0_खण्डम्[vec_अन्तः]), [ps] "r"(&scalar), [d] "r"(&dst_खण्डम्[vec_अन्तः]), [tm] "r"(पुच्छ_m0) \
                : "f10", "f11", "f12", "memory");                                                                    \
        }                                                                                                            \
                                                                                                                     \
        __asm__ चञ्चल("mova.m.x %0" ::"r"(अस्थायी_आवरणम्));                                                            \
    }

DEFINE_खण्डम्_OP_प्रसारणम्(खण्डम्_mul_प्रसारणम्, "fmul.ps")
DEFINE_खण्डम्_OP_प्रसारणम्(खण्डम्_सङ्कलय_प्रसारणम्, "fadd.ps")
DEFINE_खण्डम्_OP_प्रसारणम्(खण्डम्_sub_प्रसारणम्, "fsub.ps")

स्थिर अन्तःस्थ भिन्न scalar_el_map(भिन्न src0, भिन्न src1, गणना ggml_op operation) {
    विकल्प (operation) {
        पक्ष GGML_OP_MUL:
            प्रत्यागम src0 * src1;
        पक्ष GGML_OP_सङ्कलय:
            प्रत्यागम src0 + src1;
        पक्ष GGML_OP_SUB:
            प्रत्यागम src0 - src1;
        स्वभाव:
            प्रत्यागम 0.0f;
    }
}

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_द्विमान_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_आवरणम्);
    पूर्णाङ्क अङ्कः_तन्तवः = get_अङ्कः_तन्तवः(kernel_env->shire_आवरणम्);

    यदि (thread_id < 0) {
        प्रत्यागम 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;  // Invalid pointer
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * src1 = &params->src1;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;  // Unsupported type combination
    }

    भिन्न * src0_दत्त = (भिन्न *) src0->दत्त;
    भिन्न * src1_दत्त = (भिन्न *) src1->दत्त;
    भिन्न * dst_दत्त  = (भिन्न *) dst->दत्त;

    यदि (!src0_दत्त || !src1_दत्त || !dst_दत्त) {
        प्रत्यागम -1;  // Null data pointer
    }

#यदि_परिभाषितम् ET_UBERKERNEL
    // Consumer-side input eviction. Required because ET caches are
    // incoherent across minions: if a previous kernel in this UK batch
    // left stale lines for these addresses in this hart's L1, drop them
    // so we read fresh from L3/DRAM (where the producer flushed its
    // results). Standalone launches don't need this -- the host-side
    // runtime boundary between kernel launches handles it.
    ध्रुव size_t src0_अष्टकानि = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    ध्रुव size_t src1_अष्टकानि = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_दत्त, src0_अष्टकानि);
    evict_region_past_l2(src1_दत्त, src1_अष्टकानि);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#यदि_अन्तः

    गणना ggml_op operation = dst->op;

    यदि (operation != GGML_OP_MUL && operation != GGML_OP_सङ्कलय && operation != GGML_OP_SUB) {
        प्रत्यागम -1;  // Unsupported operation
    }

    ध्रुव int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];
    ध्रुव int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    ध्रुव int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];

    ध्रुव size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ध्रुव size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ध्रुव size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    ध्रुव bool cache_aligned = (dst->ne[0] % 16 == 0);

    // Fast path: no broadcasting, contiguous
    ध्रुव bool no_प्रसारणम् = (ne10 == ne0 && ne11 == ne1 && ne12 == ne2 && ne13 == ne3);
    ध्रुव bool all_contiguous =
        (nb0 == 4 && nb00 == 4 && nb10 == 4 && nb1 == ne0 * 4 && nb01 == ne0 * 4 && nb11 == ne0 * 4);

    यदि (no_प्रसारणम् && all_contiguous) {
        ध्रुव int64_t योग_elements         = ne0 * ne1 * ne2 * ne3;
        ध्रुव int64_t elements_प्रत्येकम्_cacheline = 16;  // 64 bytes / 4 bytes
        ध्रुव int64_t योग_cachelines       = (योग_elements + elements_प्रत्येकम्_cacheline - 1) / elements_प्रत्येकम्_cacheline;

        ध्रुव int64_t cl_प्रत्येकम्_thread = (योग_cachelines + अङ्कः_तन्तवः - 1) / अङ्कः_तन्तवः;
        ध्रुव int64_t cl_start      = thread_id * cl_प्रत्येकम्_thread;
        int64_t       cl_अन्तः        = cl_start + cl_प्रत्येकम्_thread;
        यदि (cl_अन्तः > योग_cachelines) {
            cl_अन्तः = योग_cachelines;
        }

        यदि (cl_start >= योग_cachelines) {
            प्रत्यागम 0;
        }

        ध्रुव int64_t elem_start = cl_start * elements_प्रत्येकम्_cacheline;
        int64_t       elem_अन्तः   = cl_अन्तः * elements_प्रत्येकम्_cacheline;
        यदि (elem_अन्तः > योग_elements) {
            elem_अन्तः = योग_elements;
        }
        ध्रुव int32_t गणनम् = (int32_t) (elem_अन्तः - elem_start);

        विकल्प (operation) {
            पक्ष GGML_OP_MUL:
                खण्डम्_mul_cache_aligned(dst_दत्त + elem_start, src0_दत्त + elem_start, src1_दत्त + elem_start, गणनम्);
                भङ्ग;
            पक्ष GGML_OP_सङ्कलय:
                खण्डम्_सङ्कलय_cache_aligned(dst_दत्त + elem_start, src0_दत्त + elem_start, src1_दत्त + elem_start, गणनम्);
                भङ्ग;
            पक्ष GGML_OP_SUB:
                खण्डम्_sub_cache_aligned(dst_दत्त + elem_start, src0_दत्त + elem_start, src1_दत्त + elem_start, गणनम्);
                भङ्ग;
            स्वभाव:
                प्रत्यागम 1;
        }
#यदि_परिभाषितम् ET_UBERKERNEL
        // Producer-side flush: ET caches are incoherent across minions, so
        // a consumer kernel running on a different minion can't see our
        // dirty L1 lines via its own evict_region_past_l2. Push our writes
        // all the way to DRAM so the next batched kernel reads fresh.
        // Standalone launches don't need this -- the host runtime boundary
        // between kernel launches handles cache writeback.
        FENCE;
        evict_region_past_l2(dst_दत्त + elem_start, (size_t) गणनम् * परिमाण(भिन्न));
        WAIT_CACHEOPS;
        FENCE;
#यदि_अन्तः
        प्रत्यागम 0;
    }

    // Slow path: broadcasting or non-contiguous
    ध्रुव int64_t योग_पङ्क्तिकाः = ne1 * ne2 * ne3;

    int64_t start_row;
    int64_t अन्तः_row;

    यदि (cache_aligned) {
        ध्रुव int64_t पङ्क्तिकाः_प्रत्येकम्_thread = (योग_पङ्क्तिकाः + अङ्कः_तन्तवः - 1) / अङ्कः_तन्तवः;
        start_row                     = thread_id * पङ्क्तिकाः_प्रत्येकम्_thread;
        अन्तः_row = (start_row + पङ्क्तिकाः_प्रत्येकम्_thread < योग_पङ्क्तिकाः) ? (start_row + पङ्क्तिकाः_प्रत्येकम्_thread) : योग_पङ्क्तिकाः;
    } अन्यथा {
        ध्रुव int64_t पङ्क्तिकाः_प्रत्येकम्_गणः = et_पङ्क्तिकाः_प्रत्येकम्_cacheline_गणः(ne0, परिमाण(भिन्न));
        ध्रुव int64_t योग_groups   = (योग_पङ्क्तिकाः + पङ्क्तिकाः_प्रत्येकम्_गणः - 1) / पङ्क्तिकाः_प्रत्येकम्_गणः;

        यदि (thread_id >= योग_groups) {
            प्रत्यागम 0;
        }

        ध्रुव int64_t गणः_start = thread_id;
        प्रति (int64_t grp = गणः_start; grp < योग_groups; grp += अङ्कः_तन्तवः) {
            ध्रुव int64_t गणः_row_start = grp * पङ्क्तिकाः_प्रत्येकम्_गणः;
            int64_t       गणः_row_अन्तः   = गणः_row_start + पङ्क्तिकाः_प्रत्येकम्_गणः;
            यदि (गणः_row_अन्तः > योग_पङ्क्तिकाः) {
                गणः_row_अन्तः = योग_पङ्क्तिकाः;
            }

#यदि_परिभाषितम् ET_UBERKERNEL
            // First row written by this group (used for producer-side evict).
            ध्रुव int64_t first_i03      = गणः_row_start / (ne2 * ne1);
            ध्रुव int64_t first_i02      = (गणः_row_start - first_i03 * ne2 * ne1) / ne1;
            ध्रुव int64_t first_i01      = (गणः_row_start - first_i03 * ne2 * ne1 - first_i02 * ne1);
            वर्ण *        गणः_dst_आधारः = (वर्ण *) dst_दत्त + first_i03 * nb3 + first_i02 * nb2 + first_i01 * nb1;
#यदि_अन्तः

            प्रति (int64_t ir = गणः_row_start; ir < गणः_row_अन्तः; ir++) {
                ध्रुव int64_t i03 = ir / (ne2 * ne1);
                ध्रुव int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
                ध्रुव int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

                ध्रुव int64_t i13 = i03 % ne13;
                ध्रुव int64_t i12 = i02 % ne12;
                ध्रुव int64_t i11 = i01 % ne11;

                भिन्न *       dst_ptr = (भिन्न *) ((वर्ण *) dst_दत्त + i03 * nb3 + i02 * nb2 + i01 * nb1);
                ध्रुव भिन्न * src0_ptr =
                    (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_दत्त + i03 * nb03 + i02 * nb02 + i01 * nb01);
                ध्रुव भिन्न * src1_ptr =
                    (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1_दत्त + i13 * nb13 + i12 * nb12 + i11 * nb11);

                यदि (ne10 == 1) {
                    ध्रुव भिन्न scalar = src1_ptr[0];
                    प्रति (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], scalar, operation);
                    }
                } अन्यथा {
                    प्रति (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_ptr[i0] = scalar_el_map(src0_ptr[i0], src1_ptr[i0 % ne10], operation);
                    }
                }
            }

#यदि_परिभाषितम् ET_UBERKERNEL
            // Producer-side flush for this group's rows. Group rows are
            // contiguous because nb1 = ne0*4 in the cacheline-group layout.
            // Only needed inside a UK batch; see comment in fast path.
            ध्रुव int64_t nrows = गणः_row_अन्तः - गणः_row_start;
            यदि (nrows > 0) {
                FENCE;
                evict_region_past_l2(गणः_dst_आधारः, (size_t) nrows * nb1);
                WAIT_CACHEOPS;
                FENCE;
            }
#यदि_अन्तः
        }

        प्रत्यागम 0;
    }

    यदि (start_row >= योग_पङ्क्तिकाः) {
        प्रत्यागम 0;
    }

    प्रति (int64_t ir = start_row; ir < अन्तः_row; ir++) {
        // Convert flat row index to 3D coordinates
        ध्रुव int64_t i03 = ir / (ne2 * ne1);
        ध्रुव int64_t i02 = (ir - i03 * ne2 * ne1) / ne1;
        ध्रुव int64_t i01 = (ir - i03 * ne2 * ne1 - i02 * ne1);

        // Handle broadcasting: src1 coordinates with modulo
        ध्रुव int64_t i13 = i03 % ne13;
        ध्रुव int64_t i12 = i02 % ne12;
        ध्रुव int64_t i11 = i01 % ne11;

        // Calculate base pointers for this row using stride-based addressing
        भिन्न *       dst_ptr  = (भिन्न *) ((वर्ण *) dst_दत्त + i03 * nb3 + i02 * nb2 + i01 * nb1);
        ध्रुव भिन्न * src0_ptr = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_दत्त + i03 * nb03 + i02 * nb02 + i01 * nb01);
        ध्रुव भिन्न * src1_ptr = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1_दत्त + i13 * nb13 + i12 * nb12 + i11 * nb11);

        यदि (ne10 == 1) {
            // Broadcast scalar: src1 has ne[0]=1, broadcast across entire row
            भिन्न scalar = src1_ptr[0];
            विकल्प (operation) {
                पक्ष GGML_OP_MUL:
                    खण्डम्_mul_प्रसारणम्(dst_ptr, src0_ptr, scalar, (पूर्णाङ्क) ne0);
                    भङ्ग;
                पक्ष GGML_OP_सङ्कलय:
                    खण्डम्_सङ्कलय_प्रसारणम्(dst_ptr, src0_ptr, scalar, (पूर्णाङ्क) ne0);
                    भङ्ग;
                पक्ष GGML_OP_SUB:
                    खण्डम्_sub_प्रसारणम्(dst_ptr, src0_ptr, scalar, (पूर्णाङ्क) ne0);
                    भङ्ग;
                स्वभाव:
                    प्रत्यागम 1;
            }
        } अन्यथा {
            // Broadcasting in dimension 0: src1 repeats across src0
            ध्रुव int64_t nr0 = ne0 / ne10;

            प्रति (int64_t r = 0; r < nr0; r++) {
                ध्रुव भिन्न * src0_खण्डम् = src0_ptr + r * ne10;
                भिन्न *       dst_खण्डम्  = dst_ptr + r * ne10;

                विकल्प (operation) {
                    पक्ष GGML_OP_MUL:
                        खण्डम्_mul_cache_aligned(dst_खण्डम्, src0_खण्डम्, src1_ptr, (पूर्णाङ्क) ne10);
                        भङ्ग;
                    पक्ष GGML_OP_सङ्कलय:
                        खण्डम्_सङ्कलय_cache_aligned(dst_खण्डम्, src0_खण्डम्, src1_ptr, (पूर्णाङ्क) ne10);
                        भङ्ग;
                    पक्ष GGML_OP_SUB:
                        खण्डम्_sub_cache_aligned(dst_खण्डम्, src0_खण्डम्, src1_ptr, (पूर्णाङ्क) ne10);
                        भङ्ग;
                    स्वभाव:
                        प्रत्यागम 1;
                }
            }
        }
    }

#यदि_परिभाषितम् ET_UBERKERNEL
    // Producer-side flush for the cache-aligned slow path. Rows
    // [start_row, end_row) are contiguous in dst because nb1 = ne0 * 4.
    // Only needed inside a UK batch; see comment in fast path.
    यदि (अन्तः_row > start_row) {
        FENCE;
        evict_region_past_l2((वर्ण *) dst_दत्त + start_row * nb1, (size_t) (अन्तः_row - start_row) * nb1);
        WAIT_CACHEOPS;
        FENCE;
    }
#यदि_अन्तः
    प्रत्यागम 0;
}
