//******************************************************************************
// Norm F32 Kernel (Layer Normalization)
// y[i] = (x[i] - mean) / sqrt(variance + eps)
// where mean = sum(x) / N, variance = sum((x - mean)^2) / N
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <string.h>

// Norm kernel parameters structure
ಸಂರಚನಾ ggml_et_norm_params {
    ಸಂರಚನಾ ggml_tensor src0;  // F32 input tensor
    ಸಂರಚನಾ ggml_tensor dst;   // F32 output tensor
    ಭಿನ್ನ              eps;   // Epsilon parameter for numerical stability
};

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_norm_params * params, ಶೂನ್ಯ * env) {
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
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;
    ಭಿನ್ನ                eps  = params->eps;

    ಯದಿ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Unsupported type combination
    }

    ಭಿನ್ನ * src0_data = (ಭಿನ್ನ *) src0->data;
    ಭಿನ್ನ * dst_data  = (ಭಿನ್ನ *) dst->data;

    ಯದಿ (!src0_data || !dst_data) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Null data pointer
    }


    ಯದಿ (eps < 0.0f) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Invalid epsilon
    }

    ಧ್ರುವ int64_t ne0 = dst->ne[0];
    ಧ್ರುವ int64_t ne1 = dst->ne[1];
    ಧ್ರುವ int64_t ne2 = dst->ne[2];
    ಧ್ರುವ int64_t ne3 = dst->ne[3];

    ಧ್ರುವ size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ಧ್ರುವ size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    ಯದಿ (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Shape mismatch
    }

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

                    // Step 1: sum for mean
                    ಭಿನ್ನ zero = 0.0f;
                    __asm__ ಚಞ್ಚಲ("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                    ಪ್ರತಿ (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ ಚಞ್ಚಲ(
                            "flw.ps f11, %[x_vec]\n"
                            "fadd.ps f10, f10, f11\n"
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

                    ಧ್ರುವ ಭಿನ್ನ mean = et_fdiv(sum, (ಭಿನ್ನ) (int32_t) ne0);

                    // Step 2: compute (x - mean) → dst, accumulate variance
                    __asm__ ಚಞ್ಚಲ("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                    ಪ್ರತಿ (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ ಚಞ್ಚಲ(
                            "flw.ps f11, %[x_vec]\n"
                            "fbc.ps f12, %[mean_ptr]\n"
                            "fsub.ps f13, f11, f12\n"
                            "fsw.ps f13, %[result]\n"
                            "fmadd.ps f10, f13, f13, f10\n"
                            : [result] "=m"(*(ಭಿನ್ನ (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & src_ptr[i0]), [mean_ptr] "m"(mean)
                            : "f10", "f11", "f12", "f13");
                    }

                    ಭಿನ್ನ var_sum;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(var_sum)::"t0", "f1", "f2", "f3", "f4", "f5");

                    ಧ್ರುವ ಭಿನ್ನ variance = et_fdiv(var_sum, (ಭಿನ್ನ) (int32_t) ne0);
                    ಧ್ರುವ ಭಿನ್ನ scale    = et_powf(variance + eps, -0.5f);

                    ಯದಿ (!(scale > 0.0f)) {
                        ಪ್ರತ್ಯಾಗಮ -1;
                    }

                    // Step 3: apply scale to centered values in dst
                    ಪ್ರತಿ (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ ಚಞ್ಚಲ(
                            "flw.ps f12, %[y_vec]\n"
                            "fbc.ps f13, %[scale_ptr]\n"
                            "fmul.ps f14, f12, f13\n"
                            "fsw.ps f14, %[result]\n"
                            : [result] "=m"(*(ಭಿನ್ನ (*)[8]) & dst_ptr[i0])
                            : [y_vec] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & dst_ptr[i0]), [scale_ptr] "m"(scale)
                            : "f12", "f13", "f14");
                    }
                }
            }
        }
    } ಅನ್ಯಥಾ {
        // Intra-row: threads within each shire cooperate via L2 SCP.
        // Two reductions needed: sum (for mean), then variance sum.
        ಪೂರ್ಣಾಙ್ಕ shire_tid       = thread_id % shire_threads;
        ಪೂರ್ಣಾಙ್ಕ threads_per_row = shire_threads / total_rows;
        ಪೂರ್ಣಾಙ್ಕ my_row          = shire_tid / threads_per_row;
        ಪೂರ್ಣಾಙ್ಕ local_tid       = shire_tid % threads_per_row;
        ಪೂರ್ಣಾಙ್ಕ group_base      = my_row * threads_per_row;

        ಯದಿ (my_row >= total_rows) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            // Second barrier for variance exchange
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            ಪ್ರತ್ಯಾಗಮ 0;
        }

        int64_t i1 = my_row % ne1;
        int64_t i2 = (my_row / ne1) % ne2;
        int64_t i3 = my_row / (ne1 * ne2);

        ಧ್ರುವ ಭಿನ್ನ * src_ptr = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
        ಭಿನ್ನ *       dst_ptr = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

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

        ಪೂರ್ಣಾಙ್ಕ workers = threads_per_row < total_cls ? threads_per_row : total_cls;

        unsigned long saved_mask;
        __asm__ ಚಞ್ಚಲ("mova.x.m %0" : "=r"(saved_mask));
        __asm__ ಚಞ್ಚಲ("mov.m.x m0, x0, 0xFF");

        // ---- Reduction 1: partial sum for mean ----
        __asm__ ಚಞ್ಚಲ("fbci.pi f10, 0" ::: "f10");
        ಪ್ರತಿ (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
            __asm__ ಚಞ್ಚಲ(
                "flw.ps f11, %[x_vec]\n"
                "fadd.ps f10, f10, f11\n"
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

        // L2SCP exchange for sum
        ಚಞ್ಚಲ ಭಿನ್ನ * my_slot = (ಚಞ್ಚಲ ಭಿನ್ನ *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_sum;
        FENCE;
        evict_to_l2((ಧ್ರುವ ಶೂನ್ಯ *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // All threads read sum, compute mean
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

        ಧ್ರುವ ಭಿನ್ನ mean = et_fdiv(total_sum, (ಭಿನ್ನ) (int32_t) ne0);

        // ---- Reduction 2: compute (x - mean) → dst chunk, partial variance ----
        __asm__ ಚಞ್ಚಲ("fbci.pi f10, 0" ::: "f10");

        ಯದಿ (my_start < my_end) {
            uint32_t mean_bits;
            __asm__ ಚಞ್ಚಲ("fmv.x.s %0, %1" : "=r"(mean_bits) : "f"(mean));
            __asm__ ಚಞ್ಚಲ("fbcx.ps f15, %[mb]\n" : : [mb] "r"(mean_bits) : "f15");

            ಪ್ರತಿ (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                __asm__ ಚಞ್ಚಲ(
                    "flw.ps f11, %[x_vec]\n"
                    "fsub.ps f13, f11, f15\n"
                    "fsw.ps f13, %[result]\n"
                    "fmadd.ps f10, f13, f13, f10\n"
                    : [result] "=m"(*(ಭಿನ್ನ (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & src_ptr[i0])
                    : "f10", "f11", "f13");
            }
        }

        ಭಿನ್ನ partial_var;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(partial_var)::"t0", "f1", "f2", "f3", "f4", "f5");

        // L2SCP exchange for variance (reuse same slots)
        *my_slot = partial_var;
        FENCE;
        evict_to_l2((ಧ್ರುವ ಶೂನ್ಯ *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // All threads read variance, compute scale, apply to own chunk
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ t = 0; t < workers; t++) {
            ಚಞ್ಚಲ ಭಿನ್ನ * slot = (ಚಞ್ಚಲ ಭಿನ್ನ *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            evict_to_l2((ಧ್ರುವ ಶೂನ್ಯ *) slot, 1, 64);
        }
        WAIT_CACHEOPS;

        ಭಿನ್ನ total_var = 0.0f;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ t = 0; t < workers; t++) {
            ಚಞ್ಚಲ ಭಿನ್ನ * slot = (ಚಞ್ಚಲ ಭಿನ್ನ *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            total_var += *slot;
        }

        ಧ್ರುವ ಭಿನ್ನ variance = et_fdiv(total_var, (ಭಿನ್ನ) (int32_t) ne0);
        ಧ್ರುವ ಭಿನ್ನ scale    = et_powf(variance + eps, -0.5f);

        ಯದಿ (!(scale > 0.0f)) {
            __asm__ ಚಞ್ಚಲ("mova.m.x %0" ::"r"(saved_mask));
            ಪ್ರತ್ಯಾಗಮ -1;
        }

        // Apply scale to centered values (already in dst from reduction 2)
        ಯದಿ (my_start < my_end) {
            uint32_t scale_bits;
            __asm__ ಚಞ್ಚಲ("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
            __asm__ ಚಞ್ಚಲ("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

            ಪ್ರತಿ (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                __asm__ ಚಞ್ಚಲ(
                    "flw.ps f12, %[y_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fsw.ps f14, %[result]\n"
                    : [result] "=m"(*(ಭಿನ್ನ (*)[8]) & dst_ptr[i0])
                    : [y_vec] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & dst_ptr[i0])
                    : "f12", "f14");
            }
        }

        __asm__ ಚಞ್ಚಲ("mova.m.x %0" ::"r"(saved_mask));
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}
