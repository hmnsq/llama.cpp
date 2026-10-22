
// Fused RMS Norm + MUL F32 Kernel

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "math_fp.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <assert.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <string.h>

// Fused RMS norm + MUL kernel parameters structure
संरचना ggml_et_rms_norm_mul_params {
    संरचना ggml_tensor src0;  // F32 input tensor (to be normalized)
    संरचना ggml_tensor src1;  // F32 weights tensor (element-wise multiply)
    संरचना ggml_tensor dst;   // F32 output tensor
    भिन्न              eps;   // Epsilon for numerical stability
};

स्थिर अन्तःस्थ size_t tensor_अष्टकानि(ध्रुव संरचना ggml_tensor * t) {
    प्रत्यागम (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_rms_norm_mul_params * params, शून्य * env) {
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

    भिन्न eps = params->eps;

    यदि (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;  // Unsupported type combination
    }

    भिन्न * src0_दत्त = (भिन्न *) src0->दत्त;
    भिन्न * src1_दत्त = (भिन्न *) src1->दत्त;
    भिन्न * dst_दत्त  = (भिन्न *) dst->दत्त;
    // #ifdef ET_UBERKERNEL
    //     evict_region_past_l2(src0_data, tensor_bytes(src0));
    //     evict_region_past_l2(src1_data, tensor_bytes(src1));
    //     // WAIT_CACHEOPS;
    //     FENCE;
    //     // et_barrier(ET_BARRIER_GLOBAL);
    // #endif
    यदि (!src0_दत्त || !src1_दत्त || !dst_दत्त) {
        प्रत्यागम -1;  // Null data pointer
    }

    यदि (eps < 0.0f) {
        प्रत्यागम -1;  // Invalid epsilon
    }

    ध्रुव int64_t ne0 = dst->ne[0];  // Inner dimension (row size)
    ध्रुव int64_t ne1 = dst->ne[1];  // Dimension 1
    ध्रुव int64_t ne2 = dst->ne[2];  // Dimension 2
    ध्रुव int64_t ne3 = dst->ne[3];  // Dimension 3

    // Get dst strides (in bytes)
    ध्रुव size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // Get src0 strides (in bytes)
    ध्रुव size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Get src1 (weights) strides (in bytes), supports broadcasting in dims 1,2,3
    ध्रुव size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    // Verify that src0 and dst have same shape (required for RMS norm)
    यदि (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        प्रत्यागम -1;  // Shape mismatch
    }
    // et_barrier(ET_BARRIER_GLOBAL);

    ध्रुव भिन्न   inv_ne0       = et_fdiv(1.0f, (भिन्न) (int32_t) ne0);
    ध्रुव int32_t योग_पङ्क्तिकाः    = (int32_t) (ne1 * ne2 * ne3);
    ध्रुव पूर्णाङ्क     shire_तन्तवः = SOC_MINIONS_प्रत्येकम्_SHIRE * अङ्कः_HARTS_प्रत्येकम्_MINION;

    यदि (योग_पङ्क्तिकाः >= shire_तन्तवः) {
        // Row-parallel: each thread processes whole rows
        प्रति (int64_t i3 = 0; i3 < ne3; i3++) {
            प्रति (int64_t i2 = 0; i2 < ne2; i2++) {
                प्रति (int64_t i1 = thread_id; i1 < ne1; i1 += अङ्कः_तन्तवः) {
                    ध्रुव भिन्न * src_ptr =
                        (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_दत्त + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    भिन्न * dst_ptr = (भिन्न *) ((वर्ण *) dst_दत्त + i3 * nb3 + i2 * nb2 + i1 * nb1);

                    ध्रुव भिन्न * wgt_ptr = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1_दत्त + (i3 % src1->ne[3]) * nb13 +
                                                             (i2 % src1->ne[2]) * nb12 + (i1 % src1->ne[1]) * nb11);

                    अचिह्नित दीर्घ saved_आवरणम्;
                    __asm__ चञ्चल("mova.x.m %0" : "=r"(saved_आवरणम्));
                    __asm__ चञ्चल("mov.m.x m0, x0, 0xFF");

                    // Sum of squares
                    __asm__ चञ्चल("fbci.pi f10, 0" ::: "f10");
                    प्रति (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ चञ्चल(
                            "flw.ps f11, %[x_vec]\n"
                            "fmadd.ps f10, f11, f11, f10\n"
                            :
                            : [x_vec] "m"(*(ध्रुव भिन्न (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    भिन्न योग_2;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(योग_2)::"t0", "f1", "f2", "f3", "f4", "f5");

                    ध्रुव भिन्न scale = et_powf(योग_2 * inv_ne0 + eps, -0.5f);
                    यदि (!(scale > 0.0f)) {
                        __asm__ चञ्चल("mova.m.x %0" ::"r"(saved_आवरणम्));
                        प्रत्यागम -1;
                    }

                    uint32_t scale_द्वयङ्काः;
                    __asm__ चञ्चल("fmv.x.s %0, %1" : "=r"(scale_द्वयङ्काः) : "f"(scale));
                    __asm__ चञ्चल("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_द्वयङ्काः) : "f13");

                    प्रति (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ चञ्चल(
                            "flw.ps f12, %[x_vec]\n"
                            "flw.ps f15, %[w_vec]\n"
                            "fmul.ps f14, f12, f13\n"
                            "fmul.ps f14, f14, f15\n"
                            "fsw.ps f14, %[result]\n"
                            : [परिणाम] "=m"(*(भिन्न (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(ध्रुव भिन्न (*)[8]) & src_ptr[i0]), [w_vec] "m"(*(ध्रुव भिन्न (*)[8]) &
                                                                                            wgt_ptr[i0])
                            : "f12", "f14", "f15");
                    }
                    // #ifdef ET_UBERKERNEL
                    //                 FENCE;
                    //                 evict_region_past_l2(dst_ptr, (size_t)ne0 * sizeof(float));
                    //                 WAIT_CACHEOPS;
                    //                 FENCE;
                    // #endif
                    __asm__ चञ्चल("mova.m.x %0" ::"r"(saved_आवरणम्));
                }
            }
        }
    } अन्यथा {
        // Intra-row: threads within each shire cooperate on rows via L2 SCP.
        // L2 SCP + barrier are shire-local, so use shire-local thread index.
        पूर्णाङ्क shire_tid       = thread_id % shire_तन्तवः;
        पूर्णाङ्क तन्तवः_प्रत्येकम्_row = shire_तन्तवः / योग_पङ्क्तिकाः;
        पूर्णाङ्क my_row          = shire_tid / तन्तवः_प्रत्येकम्_row;
        पूर्णाङ्क local_tid       = shire_tid % तन्तवः_प्रत्येकम्_row;
        पूर्णाङ्क गणः_आधारः      = my_row * तन्तवः_प्रत्येकम्_row;

        // Excess threads within this shire
        यदि (my_row >= योग_पङ्क्तिकाः) {
            __asm__ __volatile__("fence\n" ::: "memory");
            et_barrier(ET_BARRIER_SHIRE);
            प्रत्यागम 0;
        }

        // Unflatten row index
        int64_t i1 = my_row % ne1;
        int64_t i2 = (my_row / ne1) % ne2;
        int64_t i3 = my_row / (ne1 * ne2);

        ध्रुव भिन्न * src_ptr = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_दत्त + i3 * nb03 + i2 * nb02 + i1 * nb01);
        भिन्न *       dst_ptr = (भिन्न *) ((वर्ण *) dst_दत्त + i3 * nb3 + i2 * nb2 + i1 * nb1);

        ध्रुव भिन्न * wgt_ptr = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1_दत्त + (i3 % src1->ne[3]) * nb13 +
                                                 (i2 % src1->ne[2]) * nb12 + (i1 % src1->ne[1]) * nb11);

        // Chunk boundaries aligned to 16 floats (64-byte cache line)
        ध्रुव int32_t elems_प्रत्येकम्_cl   = 16;
        int32_t       योग_cls      = ((int32_t) ne0 + elems_प्रत्येकम्_cl - 1) / elems_प्रत्येकम्_cl;
        int32_t       cls_प्रत्येकम्_thread = (योग_cls + तन्तवः_प्रत्येकम्_row - 1) / तन्तवः_प्रत्येकम्_row;
        int32_t       my_start       = local_tid * cls_प्रत्येकम्_thread * elems_प्रत्येकम्_cl;
        int32_t       my_अन्तः         = my_start + cls_प्रत्येकम्_thread * elems_प्रत्येकम्_cl;
        यदि (my_अन्तः > (int32_t) ne0) {
            my_अन्तः = (int32_t) ne0;
        }
        यदि (my_start >= (int32_t) ne0) {
            my_start = 0;
            my_अन्तः   = 0;
        }

        अचिह्नित दीर्घ saved_आवरणम्;
        __asm__ चञ्चल("mova.x.m %0" : "=r"(saved_आवरणम्));
        __asm__ चञ्चल("mov.m.x m0, x0, 0xFF");

        // Phase 1: partial sum of squares on own chunk
        __asm__ चञ्चल("fbci.pi f10, 0" ::: "f10");
        प्रति (int32_t i0 = my_start; i0 < my_अन्तः; i0 += 8) {
            __asm__ चञ्चल(
                "flw.ps f11, %[x_vec]\n"
                "fmadd.ps f10, f11, f11, f10\n"
                :
                : [x_vec] "m"(*(ध्रुव भिन्न (*)[8]) & src_ptr[i0])
                : "f10", "f11");
        }

        भिन्न partial_योग;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(partial_योग)::"t0", "f1", "f2", "f3", "f4", "f5");

        // Phase 2: write partial sum to L2 SCP, evict from L1D
        चञ्चल भिन्न * my_slot = (चञ्चल भिन्न *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_योग;
        __asm__ __volatile__("fence\n" ::: "memory");
        evict_गन्तव्यम्_l2((ध्रुव शून्य *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // Phase 3: all threads read partial sums, compute scale, apply to own chunk
        पूर्णाङ्क workers = तन्तवः_प्रत्येकम्_row < योग_cls ? तन्तवः_प्रत्येकम्_row : योग_cls;

        प्रति (पूर्णाङ्क t = 0; t < workers; t++) {
            चञ्चल भिन्न * slot = (चञ्चल भिन्न *) et_shire_l2scp_local((uint64_t) (गणः_आधारः + t) * 64);
            evict_गन्तव्यम्_l2((ध्रुव शून्य *) slot, 1, 64);
        }
        WAIT_CACHEOPS;

        भिन्न योग_योग = 0.0f;
        प्रति (पूर्णाङ्क t = 0; t < workers; t++) {
            चञ्चल भिन्न * slot = (चञ्चल भिन्न *) et_shire_l2scp_local((uint64_t) (गणः_आधारः + t) * 64);
            योग_योग += *slot;
        }

        ध्रुव भिन्न scale = et_powf(योग_योग * inv_ne0 + eps, -0.5f);
        यदि (!(scale > 0.0f)) {
            __asm__ चञ्चल("mova.m.x %0" ::"r"(saved_आवरणम्));
            प्रत्यागम -1;
        }

        // Apply scale * weights to own chunk
        यदि (my_start < my_अन्तः) {
            uint32_t scale_द्वयङ्काः;
            __asm__ चञ्चल("fmv.x.s %0, %1" : "=r"(scale_द्वयङ्काः) : "f"(scale));
            __asm__ चञ्चल("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_द्वयङ्काः) : "f13");

            प्रति (int32_t i0 = my_start; i0 < my_अन्तः; i0 += 8) {
                __asm__ चञ्चल(
                    "flw.ps f12, %[x_vec]\n"
                    "flw.ps f15, %[w_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fmul.ps f14, f14, f15\n"
                    "fsw.ps f14, %[result]\n"
                    : [परिणाम] "=m"(*(भिन्न (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(ध्रुव भिन्न (*)[8]) & src_ptr[i0]), [w_vec] "m"(*(ध्रुव भिन्न (*)[8]) & wgt_ptr[i0])
                    : "f12", "f14", "f15");
            }
            // #ifdef ET_UBERKERNEL
            //             FENCE;
            //             evict_region_past_l2(dst_ptr + my_start, (size_t)(my_end - my_start) * sizeof(float));
            //             WAIT_CACHEOPS;
            //             FENCE;
            // #endif
        }

        __asm__ चञ्चल("mova.m.x %0" ::"r"(saved_आवरणम्));
    }

    प्रत्यागम 0;
}
