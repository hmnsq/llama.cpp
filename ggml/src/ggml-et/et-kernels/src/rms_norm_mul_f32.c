
// Fused RMS Norm + MUL F32 Kernel

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <string.h>

// Fused RMS norm + MUL kernel parameters structure
ಸಂರಚನಾ ggml_et_rms_norm_mul_params {
    ಸಂರಚನಾ ggml_tensor src0;  // F32 input tensor (to be normalized)
    ಸಂರಚನಾ ggml_tensor src1;  // F32 weights tensor (element-wise multiply)
    ಸಂರಚನಾ ggml_tensor dst;   // F32 output tensor
    ಭಿನ್ನ              eps;   // Epsilon for numerical stability
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t tensor_bytes(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * t) {
    ಪ್ರತ್ಯಾಗಮ (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_rms_norm_mul_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಯದಿ (!kernel_env) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪೂರ್ಣಾಙ್ಕ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_threads = get_ಅಂಕೆ_threads(kernel_env->shire_mask);

    ಯದಿ (thread_id < 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಯದಿ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Invalid pointer
    }

    ಸಂರಚನಾ ggml_tensor * src0 = &params->src0;
    ಸಂರಚನಾ ggml_tensor * src1 = &params->src1;
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;

    ಭಿನ್ನ eps = params->eps;

    ಯದಿ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Unsupported type combination
    }

    ಭಿನ್ನ * src0_data = (ಭಿನ್ನ *) src0->data;
    ಭಿನ್ನ * src1_data = (ಭಿನ್ನ *) src1->data;
    ಭಿನ್ನ * dst_data  = (ಭಿನ್ನ *) dst->data;
    // #ifdef ET_UBERKERNEL
    //     evict_region_past_l2(src0_data, tensor_bytes(src0));
    //     evict_region_past_l2(src1_data, tensor_bytes(src1));
    //     // WAIT_CACHEOPS;
    //     FENCE;
    //     // et_barrier(ET_BARRIER_GLOBAL);
    // #endif
    ಯದಿ (!src0_data || !src1_data || !dst_data) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Null data pointer
    }

    ಯದಿ (eps < 0.0f) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Invalid epsilon
    }

    ಧ್ರುವ int64_t ne0 = dst->ne[0];  // Inner dimension (row size)
    ಧ್ರುವ int64_t ne1 = dst->ne[1];  // Dimension 1
    ಧ್ರುವ int64_t ne2 = dst->ne[2];  // Dimension 2
    ಧ್ರುವ int64_t ne3 = dst->ne[3];  // Dimension 3

    // Get dst strides (in bytes)
    ಧ್ರುವ size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // Get src0 strides (in bytes)
    ಧ್ರುವ size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Get src1 (weights) strides (in bytes), supports broadcasting in dims 1,2,3
    ಧ್ರುವ size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];

    // Verify that src0 and dst have same shape (required for RMS norm)
    ಯದಿ (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Shape mismatch
    }
    // et_barrier(ET_BARRIER_GLOBAL);

    ಧ್ರುವ ಭಿನ್ನ   inv_ne0       = et_fdiv(1.0f, (ಭಿನ್ನ) (int32_t) ne0);
    ಧ್ರುವ int32_t total_rows    = (int32_t) (ne1 * ne2 * ne3);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     shire_threads = SOC_MINIONS_PER_SHIRE * ಅಂಕೆ_HARTS_PER_MINION;

    ಯದಿ (total_rows >= shire_threads) {
        // Row-parallel: each thread processes whole rows
        ಪ್ರತಿ (int64_t i3 = 0; i3 < ne3; i3++) {
            ಪ್ರತಿ (int64_t i2 = 0; i2 < ne2; i2++) {
                ಪ್ರತಿ (int64_t i1 = thread_id; i1 < ne1; i1 += ಅಂಕೆ_threads) {
                    ಧ್ರುವ ಭಿನ್ನ * src_ptr =
                        (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    ಭಿನ್ನ * dst_ptr = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

                    ಧ್ರುವ ಭಿನ್ನ * wgt_ptr = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src1_data + (i3 % src1->ne[3]) * nb13 +
                                                             (i2 % src1->ne[2]) * nb12 + (i1 % src1->ne[1]) * nb11);

                    unsigned long saved_mask;
                    __asm__ ಚಞ್ಚಲ("mova.x.m %0" : "=r"(saved_mask));
                    __asm__ ಚಞ್ಚಲ("mov.m.x m0, x0, 0xFF");

                    // Sum of squares
                    __asm__ ಚಞ್ಚಲ("fbci.pi f10, 0" ::: "f10");
                    ಪ್ರತಿ (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ ಚಞ್ಚಲ(
                            "flw.ps f11, %[x_vec]\n"
                            "fmadd.ps f10, f11, f11, f10\n"
                            :
                            : [x_vec] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    ಭಿನ್ನ sum;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(sum)::"t0", "f1", "f2", "f3", "f4", "f5");

                    ಧ್ರುವ ಭಿನ್ನ scale = et_powf(sum * inv_ne0 + eps, -0.5f);
                    ಯದಿ (!(scale > 0.0f)) {
                        __asm__ ಚಞ್ಚಲ("mova.m.x %0" ::"r"(saved_mask));
                        ಪ್ರತ್ಯಾಗಮ -1;
                    }

                    uint32_t scale_bits;
                    __asm__ ಚಞ್ಚಲ("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
                    __asm__ ಚಞ್ಚಲ("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

                    ಪ್ರತಿ (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ ಚಞ್ಚಲ(
                            "flw.ps f12, %[x_vec]\n"
                            "flw.ps f15, %[w_vec]\n"
                            "fmul.ps f14, f12, f13\n"
                            "fmul.ps f14, f14, f15\n"
                            "fsw.ps f14, %[result]\n"
                            : [result] "=m"(*(ಭಿನ್ನ (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & src_ptr[i0]), [w_vec] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) &
                                                                                            wgt_ptr[i0])
                            : "f12", "f14", "f15");
                    }
                    // #ifdef ET_UBERKERNEL
                    //                 FENCE;
                    //                 evict_region_past_l2(dst_ptr, (size_t)ne0 * sizeof(float));
                    //                 WAIT_CACHEOPS;
                    //                 FENCE;
                    // #endif
                    __asm__ ಚಞ್ಚಲ("mova.m.x %0" ::"r"(saved_mask));
                }
            }
        }
    } ಅನ್ಯಥಾ {
        // Intra-row: threads within each shire cooperate on rows via L2 SCP.
        // L2 SCP + barrier are shire-local, so use shire-local thread index.
        ಪೂರ್ಣಾಙ್ಕ shire_tid       = thread_id % shire_threads;
        ಪೂರ್ಣಾಙ್ಕ threads_per_row = shire_threads / total_rows;
        ಪೂರ್ಣಾಙ್ಕ my_row          = shire_tid / threads_per_row;
        ಪೂರ್ಣಾಙ್ಕ local_tid       = shire_tid % threads_per_row;
        ಪೂರ್ಣಾಙ್ಕ group_base      = my_row * threads_per_row;

        // Excess threads within this shire
        ಯದಿ (my_row >= total_rows) {
            __asm__ __volatile__("fence\n" ::: "memory");
            et_barrier(ET_BARRIER_SHIRE);
            ಪ್ರತ್ಯಾಗಮ 0;
        }

        // Unflatten row index
        int64_t i1 = my_row % ne1;
        int64_t i2 = (my_row / ne1) % ne2;
        int64_t i3 = my_row / (ne1 * ne2);

        ಧ್ರುವ ಭಿನ್ನ * src_ptr = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
        ಭಿನ್ನ *       dst_ptr = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

        ಧ್ರುವ ಭಿನ್ನ * wgt_ptr = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src1_data + (i3 % src1->ne[3]) * nb13 +
                                                 (i2 % src1->ne[2]) * nb12 + (i1 % src1->ne[1]) * nb11);

        // Chunk boundaries aligned to 16 floats (64-byte cache line)
        ಧ್ರುವ int32_t elems_per_cl   = 16;
        int32_t       total_cls      = ((int32_t) ne0 + elems_per_cl - 1) / elems_per_cl;
        int32_t       cls_per_thread = (total_cls + threads_per_row - 1) / threads_per_row;
        int32_t       my_start       = local_tid * cls_per_thread * elems_per_cl;
        int32_t       my_end         = my_start + cls_per_thread * elems_per_cl;
        ಯದಿ (my_end > (int32_t) ne0) {
            my_end = (int32_t) ne0;
        }
        ಯದಿ (my_start >= (int32_t) ne0) {
            my_start = 0;
            my_end   = 0;
        }

        unsigned long saved_mask;
        __asm__ ಚಞ್ಚಲ("mova.x.m %0" : "=r"(saved_mask));
        __asm__ ಚಞ್ಚಲ("mov.m.x m0, x0, 0xFF");

        // Phase 1: partial sum of squares on own chunk
        __asm__ ಚಞ್ಚಲ("fbci.pi f10, 0" ::: "f10");
        ಪ್ರತಿ (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
            __asm__ ಚಞ್ಚಲ(
                "flw.ps f11, %[x_vec]\n"
                "fmadd.ps f10, f11, f11, f10\n"
                :
                : [x_vec] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & src_ptr[i0])
                : "f10", "f11");
        }

        ಭಿನ್ನ partial_sum;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(partial_sum)::"t0", "f1", "f2", "f3", "f4", "f5");

        // Phase 2: write partial sum to L2 SCP, evict from L1D
        ಚಞ್ಚಲ ಭಿನ್ನ * my_slot = (ಚಞ್ಚಲ ಭಿನ್ನ *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_sum;
        __asm__ __volatile__("fence\n" ::: "memory");
        evict_to_l2((ಧ್ರುವ ಶೂನ್ಯ *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // Phase 3: all threads read partial sums, compute scale, apply to own chunk
        ಪೂರ್ಣಾಙ್ಕ workers = threads_per_row < total_cls ? threads_per_row : total_cls;

        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ t = 0; t < workers; t++) {
            ಚಞ್ಚಲ ಭಿನ್ನ * slot = (ಚಞ್ಚಲ ಭಿನ್ನ *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            evict_to_l2((ಧ್ರುವ ಶೂನ್ಯ *) slot, 1, 64);
        }
        WAIT_CACHEOPS;

        ಭಿನ್ನ total_sum = 0.0f;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ t = 0; t < workers; t++) {
            ಚಞ್ಚಲ ಭಿನ್ನ * slot = (ಚಞ್ಚಲ ಭಿನ್ನ *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            total_sum += *slot;
        }

        ಧ್ರುವ ಭಿನ್ನ scale = et_powf(total_sum * inv_ne0 + eps, -0.5f);
        ಯದಿ (!(scale > 0.0f)) {
            __asm__ ಚಞ್ಚಲ("mova.m.x %0" ::"r"(saved_mask));
            ಪ್ರತ್ಯಾಗಮ -1;
        }

        // Apply scale * weights to own chunk
        ಯದಿ (my_start < my_end) {
            uint32_t scale_bits;
            __asm__ ಚಞ್ಚಲ("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
            __asm__ ಚಞ್ಚಲ("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

            ಪ್ರತಿ (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                __asm__ ಚಞ್ಚಲ(
                    "flw.ps f12, %[x_vec]\n"
                    "flw.ps f15, %[w_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fmul.ps f14, f14, f15\n"
                    "fsw.ps f14, %[result]\n"
                    : [result] "=m"(*(ಭಿನ್ನ (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & src_ptr[i0]), [w_vec] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & wgt_ptr[i0])
                    : "f12", "f14", "f15");
            }
            // #ifdef ET_UBERKERNEL
            //             FENCE;
            //             evict_region_past_l2(dst_ptr + my_start, (size_t)(my_end - my_start) * sizeof(float));
            //             WAIT_CACHEOPS;
            //             FENCE;
            // #endif
        }

        __asm__ ಚಞ್ಚಲ("mova.m.x %0" ::"r"(saved_mask));
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}
