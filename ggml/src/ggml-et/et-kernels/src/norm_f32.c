//******************************************************************************
// Norm F32 Kernel (Layer Normalization)
// y[i] = (x[i] - mean) / sqrt(variance + eps)
// where mean = sum(x) / N, variance = sum((x - mean)^2) / N
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <assert.h>
#شامل <stdint.h>
#شامل <string.h>

// Norm kernel parameters structure
ساختار ggml_et_norm_params {
    ساختار ggml_tensor src0;  // F32 input tensor
    ساختار ggml_tensor dst;   // F32 output tensor
    اعشاری              eps;   // Epsilon parameter for numerical stability
};

صحیح entry_point(ساختار ggml_et_norm_params * params, تهی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        بازگشت -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحیح رقم_threads = get_رقم_threads(kernel_env->shire_mask);

    اگر (thread_id < 0) {
        بازگشت 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        بازگشت -1;  // Invalid pointer
    }

    ساختار ggml_tensor * src0 = &params->src0;
    ساختار ggml_tensor * dst  = &params->dst;
    اعشاری                eps  = params->eps;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        بازگشت -1;  // Unsupported type combination
    }

    اعشاری * src0_data = (اعشاری *) src0->data;
    اعشاری * dst_data  = (اعشاری *) dst->data;

    اگر (!src0_data || !dst_data) {
        بازگشت -1;  // Null data pointer
    }


    اگر (eps < 0.0f) {
        بازگشت -1;  // Invalid epsilon
    }

    ثابت int64_t ne0 = dst->ne[0];
    ثابت int64_t ne1 = dst->ne[1];
    ثابت int64_t ne2 = dst->ne[2];
    ثابت int64_t ne3 = dst->ne[3];

    ثابت size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ثابت size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    اگر (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        بازگشت -1;  // Shape mismatch
    }

    ثابت int32_t total_rows    = (int32_t) (ne1 * ne2 * ne3);
    ثابت صحیح     shire_threads = SOC_MINIONS_PER_SHIRE * رقم_HARTS_PER_MINION;

    اگر (total_rows >= shire_threads) {
        // Row-parallel: each thread processes whole rows
        برای (int64_t i3 = 0; i3 < ne3; i3++) {
            برای (int64_t i2 = 0; i2 < ne2; i2++) {
                برای (int64_t i1 = thread_id; i1 < ne1; i1 += رقم_threads) {
                    ثابت اعشاری * src_ptr =
                        (ثابت اعشاری *) ((ثابت نویسه *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    اعشاری * dst_ptr = (اعشاری *) ((نویسه *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

                    // Step 1: sum for mean
                    اعشاری zero = 0.0f;
                    __asm__ فرار("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                    برای (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ فرار(
                            "flw.ps f11, %[x_vec]\n"
                            "fadd.ps f10, f10, f11\n"
                            :
                            : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    اعشاری sum;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(sum)::"t0", "f1", "f2", "f3", "f4", "f5");

                    ثابت اعشاری mean = et_fdiv(sum, (اعشاری) (int32_t) ne0);

                    // Step 2: compute (x - mean) → dst, accumulate variance
                    __asm__ فرار("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                    برای (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ فرار(
                            "flw.ps f11, %[x_vec]\n"
                            "fbc.ps f12, %[mean_ptr]\n"
                            "fsub.ps f13, f11, f12\n"
                            "fsw.ps f13, %[result]\n"
                            "fmadd.ps f10, f13, f13, f10\n"
                            : [result] "=m"(*(اعشاری (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & src_ptr[i0]), [mean_ptr] "m"(mean)
                            : "f10", "f11", "f12", "f13");
                    }

                    اعشاری var_sum;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(var_sum)::"t0", "f1", "f2", "f3", "f4", "f5");

                    ثابت اعشاری variance = et_fdiv(var_sum, (اعشاری) (int32_t) ne0);
                    ثابت اعشاری scale    = et_powf(variance + eps, -0.5f);

                    اگر (!(scale > 0.0f)) {
                        بازگشت -1;
                    }

                    // Step 3: apply scale to centered values in dst
                    برای (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ فرار(
                            "flw.ps f12, %[y_vec]\n"
                            "fbc.ps f13, %[scale_ptr]\n"
                            "fmul.ps f14, f12, f13\n"
                            "fsw.ps f14, %[result]\n"
                            : [result] "=m"(*(اعشاری (*)[8]) & dst_ptr[i0])
                            : [y_vec] "m"(*(ثابت اعشاری (*)[8]) & dst_ptr[i0]), [scale_ptr] "m"(scale)
                            : "f12", "f13", "f14");
                    }
                }
            }
        }
    } وگرنه {
        // Intra-row: threads within each shire cooperate via L2 SCP.
        // Two reductions needed: sum (for mean), then variance sum.
        صحیح shire_tid       = thread_id % shire_threads;
        صحیح threads_per_row = shire_threads / total_rows;
        صحیح my_row          = shire_tid / threads_per_row;
        صحیح local_tid       = shire_tid % threads_per_row;
        صحیح group_base      = my_row * threads_per_row;

        اگر (my_row >= total_rows) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            // Second barrier for variance exchange
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            بازگشت 0;
        }

        int64_t i1 = my_row % ne1;
        int64_t i2 = (my_row / ne1) % ne2;
        int64_t i3 = my_row / (ne1 * ne2);

        ثابت اعشاری * src_ptr = (ثابت اعشاری *) ((ثابت نویسه *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
        اعشاری *       dst_ptr = (اعشاری *) ((نویسه *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

        ثابت int32_t elems_per_cl   = 16;
        int32_t       total_cls      = ((int32_t) ne0 + elems_per_cl - 1) / elems_per_cl;
        int32_t       cls_per_thread = (total_cls + threads_per_row - 1) / threads_per_row;
        int32_t       my_start       = local_tid * cls_per_thread * elems_per_cl;
        int32_t       my_end         = my_start + cls_per_thread * elems_per_cl;
        اگر (my_end > (int32_t) ne0) {
            my_end = (int32_t) ne0;
        }
        اگر (my_start >= (int32_t) ne0) {
            my_start = 0;
            my_end   = 0;
        }

        صحیح workers = threads_per_row < total_cls ? threads_per_row : total_cls;

        unsigned long saved_mask;
        __asm__ فرار("mova.x.m %0" : "=r"(saved_mask));
        __asm__ فرار("mov.m.x m0, x0, 0xFF");

        // ---- Reduction 1: partial sum for mean ----
        __asm__ فرار("fbci.pi f10, 0" ::: "f10");
        برای (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
            __asm__ فرار(
                "flw.ps f11, %[x_vec]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & src_ptr[i0])
                : "f10", "f11");
        }

        اعشاری partial_sum;
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
        فرار اعشاری * my_slot = (فرار اعشاری *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_sum;
        FENCE;
        evict_to_l2((ثابت تهی *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // All threads read sum, compute mean
        برای (صحیح t = 0; t < workers; t++) {
            فرار اعشاری * slot = (فرار اعشاری *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            evict_to_l2((ثابت تهی *) slot, 1, 64);
        }
        WAIT_CACHEOPS;

        اعشاری total_sum = 0.0f;
        برای (صحیح t = 0; t < workers; t++) {
            فرار اعشاری * slot = (فرار اعشاری *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            total_sum += *slot;
        }

        ثابت اعشاری mean = et_fdiv(total_sum, (اعشاری) (int32_t) ne0);

        // ---- Reduction 2: compute (x - mean) → dst chunk, partial variance ----
        __asm__ فرار("fbci.pi f10, 0" ::: "f10");

        اگر (my_start < my_end) {
            uint32_t mean_bits;
            __asm__ فرار("fmv.x.s %0, %1" : "=r"(mean_bits) : "f"(mean));
            __asm__ فرار("fbcx.ps f15, %[mb]\n" : : [mb] "r"(mean_bits) : "f15");

            برای (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                __asm__ فرار(
                    "flw.ps f11, %[x_vec]\n"
                    "fsub.ps f13, f11, f15\n"
                    "fsw.ps f13, %[result]\n"
                    "fmadd.ps f10, f13, f13, f10\n"
                    : [result] "=m"(*(اعشاری (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & src_ptr[i0])
                    : "f10", "f11", "f13");
            }
        }

        اعشاری partial_var;
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
        evict_to_l2((ثابت تهی *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // All threads read variance, compute scale, apply to own chunk
        برای (صحیح t = 0; t < workers; t++) {
            فرار اعشاری * slot = (فرار اعشاری *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            evict_to_l2((ثابت تهی *) slot, 1, 64);
        }
        WAIT_CACHEOPS;

        اعشاری total_var = 0.0f;
        برای (صحیح t = 0; t < workers; t++) {
            فرار اعشاری * slot = (فرار اعشاری *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            total_var += *slot;
        }

        ثابت اعشاری variance = et_fdiv(total_var, (اعشاری) (int32_t) ne0);
        ثابت اعشاری scale    = et_powf(variance + eps, -0.5f);

        اگر (!(scale > 0.0f)) {
            __asm__ فرار("mova.m.x %0" ::"r"(saved_mask));
            بازگشت -1;
        }

        // Apply scale to centered values (already in dst from reduction 2)
        اگر (my_start < my_end) {
            uint32_t scale_bits;
            __asm__ فرار("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
            __asm__ فرار("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

            برای (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                __asm__ فرار(
                    "flw.ps f12, %[y_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fsw.ps f14, %[result]\n"
                    : [result] "=m"(*(اعشاری (*)[8]) & dst_ptr[i0])
                    : [y_vec] "m"(*(ثابت اعشاری (*)[8]) & dst_ptr[i0])
                    : "f12", "f14");
            }
        }

        __asm__ فرار("mova.m.x %0" ::"r"(saved_mask));
    }

    بازگشت 0;
}
