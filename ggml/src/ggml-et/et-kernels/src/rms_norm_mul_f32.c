
// Fused RMS Norm + MUL F32 Kernel

#समावेश "ggml_tensor.h"
#समावेश "math_fp.h"
#समावेश "platform.h"

#समावेश <assert.h>
#समावेश <stdint.h>
#समावेश <string.h>

// Fused RMS norm + MUL kernel parameters structure
काष्ठा ggml_et_rms_norm_mul_params {
    काष्ठा ggml_tensor src0;  // F32 input tensor (to be normalized)
    काष्ठा ggml_tensor src1;  // F32 weights tensor (element-wise multiply)
    काष्ठा ggml_tensor dst;   // F32 output tensor
    भग्न              eps;   // Epsilon for numerical stability
};

अटल अंतरभूत माप_प्रकार tensor_bytes(स्थिर काष्ठा ggml_tensor * t) {
    वापस (माप_प्रकार) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

पूर्णांक entry_point(काष्ठा ggml_et_rms_norm_mul_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस 0;
    }

    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;  // Invalid pointer
    }

    काष्ठा ggml_tensor * src0 = &params->src0;
    काष्ठा ggml_tensor * src1 = &params->src1;
    काष्ठा ggml_tensor * dst  = &params->dst;

    भग्न eps = params->eps;

    अगर (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस -1;  // Unsupported type combination
    }

    भग्न * src0_तथ्य = (भग्न *) src0->तथ्य;
    भग्न * src1_तथ्य = (भग्न *) src1->तथ्य;
    भग्न * dst_तथ्य  = (भग्न *) dst->तथ्य;
    // #ifdef ET_UBERKERNEL
    //     evict_region_past_l2(src0_data, tensor_bytes(src0));
    //     evict_region_past_l2(src1_data, tensor_bytes(src1));
    //     // WAIT_CACHEOPS;
    //     FENCE;
    //     // et_barrier(ET_BARRIER_GLOBAL);
    // #endif
    अगर (!src0_तथ्य || !src1_तथ्य || !dst_तथ्य) {
        वापस -1;  // Null data pointer
    }

    अगर (eps < 0.0f) {
        वापस -1;  // Invalid epsilon
    }

    स्थिर int64_t ne0 = dst->ne[0];  // Inner dimension (row size)
    स्थिर int64_t ne1 = dst->ne[1];  // Dimension 1
    स्थिर int64_t ne2 = dst->ne[2];  // Dimension 2
    स्थिर int64_t ne3 = dst->ne[3];  // Dimension 3

    // Get dst strides (in bytes)
    स्थिर माप_प्रकार nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // Get src0 strides (in bytes)
    स्थिर माप_प्रकार nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Get src1 (weights) strides (in bytes), supports broadcasting in dims 1,2,3
    स्थिर माप_प्रकार nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    // Verify that src0 and dst have same shape (required for RMS norm)
    अगर (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        वापस -1;  // Shape mismatch
    }
    // et_barrier(ET_BARRIER_GLOBAL);

    स्थिर भग्न   inv_ne0       = et_fdiv(1.0f, (भग्न) (int32_t) ne0);
    स्थिर int32_t योग_rows    = (int32_t) (ne1 * ne2 * ne3);
    स्थिर पूर्णांक     shire_threads = SOC_MINIONS_PER_SHIRE * अंक_HARTS_PER_MINION;

    अगर (योग_rows >= shire_threads) {
        // Row-parallel: each thread processes whole rows
        क्रम (int64_t i3 = 0; i3 < ne3; i3++) {
            क्रम (int64_t i2 = 0; i2 < ne2; i2++) {
                क्रम (int64_t i1 = thread_id; i1 < ne1; i1 += अंक_threads) {
                    स्थिर भग्न * src_ptr =
                        (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    भग्न * dst_ptr = (भग्न *) ((अक्षर *) dst_तथ्य + i3 * nb3 + i2 * nb2 + i1 * nb1);

                    स्थिर भग्न * wgt_ptr = (स्थिर भग्न *) ((स्थिर अक्षर *) src1_तथ्य + (i3 % src1->ne[3]) * nb13 +
                                                             (i2 % src1->ne[2]) * nb12 + (i1 % src1->ne[1]) * nb11);

                    अचिन्हित दीर्घ saved_mask;
                    __asm__ अस्थिर("mova.x.m %0" : "=r"(saved_mask));
                    __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");

                    // Sum of squares
                    __asm__ अस्थिर("fbci.pi f10, 0" ::: "f10");
                    क्रम (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ अस्थिर(
                            "flw.ps f11, %[x_vec]\n"
                            "fmadd.ps f10, f11, f11, f10\n"
                            :
                            : [x_vec] "m"(*(स्थिर भग्न (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    भग्न योग_2;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(योग_2)::"t0", "f1", "f2", "f3", "f4", "f5");

                    स्थिर भग्न scale = et_powf(योग_2 * inv_ne0 + eps, -0.5f);
                    अगर (!(scale > 0.0f)) {
                        __asm__ अस्थिर("mova.m.x %0" ::"r"(saved_mask));
                        वापस -1;
                    }

                    uint32_t scale_bits;
                    __asm__ अस्थिर("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
                    __asm__ अस्थिर("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

                    क्रम (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ अस्थिर(
                            "flw.ps f12, %[x_vec]\n"
                            "flw.ps f15, %[w_vec]\n"
                            "fmul.ps f14, f12, f13\n"
                            "fmul.ps f14, f14, f15\n"
                            "fsw.ps f14, %[result]\n"
                            : [परिणाम] "=m"(*(भग्न (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(स्थिर भग्न (*)[8]) & src_ptr[i0]), [w_vec] "m"(*(स्थिर भग्न (*)[8]) &
                                                                                            wgt_ptr[i0])
                            : "f12", "f14", "f15");
                    }
                    // #ifdef ET_UBERKERNEL
                    //                 FENCE;
                    //                 evict_region_past_l2(dst_ptr, (size_t)ne0 * sizeof(float));
                    //                 WAIT_CACHEOPS;
                    //                 FENCE;
                    // #endif
                    __asm__ अस्थिर("mova.m.x %0" ::"r"(saved_mask));
                }
            }
        }
    } अन्यथा {
        // Intra-row: threads within each shire cooperate on rows via L2 SCP.
        // L2 SCP + barrier are shire-local, so use shire-local thread index.
        पूर्णांक shire_tid       = thread_id % shire_threads;
        पूर्णांक threads_per_row = shire_threads / योग_rows;
        पूर्णांक my_row          = shire_tid / threads_per_row;
        पूर्णांक local_tid       = shire_tid % threads_per_row;
        पूर्णांक group_base      = my_row * threads_per_row;

        // Excess threads within this shire
        अगर (my_row >= योग_rows) {
            __asm__ __volatile__("fence\n" ::: "memory");
            et_barrier(ET_BARRIER_SHIRE);
            वापस 0;
        }

        // Unflatten row index
        int64_t i1 = my_row % ne1;
        int64_t i2 = (my_row / ne1) % ne2;
        int64_t i3 = my_row / (ne1 * ne2);

        स्थिर भग्न * src_ptr = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i3 * nb03 + i2 * nb02 + i1 * nb01);
        भग्न *       dst_ptr = (भग्न *) ((अक्षर *) dst_तथ्य + i3 * nb3 + i2 * nb2 + i1 * nb1);

        स्थिर भग्न * wgt_ptr = (स्थिर भग्न *) ((स्थिर अक्षर *) src1_तथ्य + (i3 % src1->ne[3]) * nb13 +
                                                 (i2 % src1->ne[2]) * nb12 + (i1 % src1->ne[1]) * nb11);

        // Chunk boundaries aligned to 16 floats (64-byte cache line)
        स्थिर int32_t elems_per_cl   = 16;
        int32_t       योग_cls      = ((int32_t) ne0 + elems_per_cl - 1) / elems_per_cl;
        int32_t       cls_per_thread = (योग_cls + threads_per_row - 1) / threads_per_row;
        int32_t       my_start       = local_tid * cls_per_thread * elems_per_cl;
        int32_t       my_end         = my_start + cls_per_thread * elems_per_cl;
        अगर (my_end > (int32_t) ne0) {
            my_end = (int32_t) ne0;
        }
        अगर (my_start >= (int32_t) ne0) {
            my_start = 0;
            my_end   = 0;
        }

        अचिन्हित दीर्घ saved_mask;
        __asm__ अस्थिर("mova.x.m %0" : "=r"(saved_mask));
        __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");

        // Phase 1: partial sum of squares on own chunk
        __asm__ अस्थिर("fbci.pi f10, 0" ::: "f10");
        क्रम (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
            __asm__ अस्थिर(
                "flw.ps f11, %[x_vec]\n"
                "fmadd.ps f10, f11, f11, f10\n"
                :
                : [x_vec] "m"(*(स्थिर भग्न (*)[8]) & src_ptr[i0])
                : "f10", "f11");
        }

        भग्न partial_योग;
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
        अस्थिर भग्न * my_slot = (अस्थिर भग्न *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_योग;
        __asm__ __volatile__("fence\n" ::: "memory");
        evict_to_l2((स्थिर व्योम *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // Phase 3: all threads read partial sums, compute scale, apply to own chunk
        पूर्णांक workers = threads_per_row < योग_cls ? threads_per_row : योग_cls;

        क्रम (पूर्णांक t = 0; t < workers; t++) {
            अस्थिर भग्न * slot = (अस्थिर भग्न *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            evict_to_l2((स्थिर व्योम *) slot, 1, 64);
        }
        WAIT_CACHEOPS;

        भग्न योग_योग = 0.0f;
        क्रम (पूर्णांक t = 0; t < workers; t++) {
            अस्थिर भग्न * slot = (अस्थिर भग्न *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            योग_योग += *slot;
        }

        स्थिर भग्न scale = et_powf(योग_योग * inv_ne0 + eps, -0.5f);
        अगर (!(scale > 0.0f)) {
            __asm__ अस्थिर("mova.m.x %0" ::"r"(saved_mask));
            वापस -1;
        }

        // Apply scale * weights to own chunk
        अगर (my_start < my_end) {
            uint32_t scale_bits;
            __asm__ अस्थिर("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
            __asm__ अस्थिर("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

            क्रम (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                __asm__ अस्थिर(
                    "flw.ps f12, %[x_vec]\n"
                    "flw.ps f15, %[w_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fmul.ps f14, f14, f15\n"
                    "fsw.ps f14, %[result]\n"
                    : [परिणाम] "=m"(*(भग्न (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(स्थिर भग्न (*)[8]) & src_ptr[i0]), [w_vec] "m"(*(स्थिर भग्न (*)[8]) & wgt_ptr[i0])
                    : "f12", "f14", "f15");
            }
            // #ifdef ET_UBERKERNEL
            //             FENCE;
            //             evict_region_past_l2(dst_ptr + my_start, (size_t)(my_end - my_start) * sizeof(float));
            //             WAIT_CACHEOPS;
            //             FENCE;
            // #endif
        }

        __asm__ अस्थिर("mova.m.x %0" ::"r"(saved_mask));
    }

    वापस 0;
}
