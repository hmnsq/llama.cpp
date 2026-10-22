//******************************************************************************
// Norm F32 Kernel (Layer Normalization)
// y[i] = (x[i] - mean) / sqrt(variance + eps)
// where mean = sum(x) / N, variance = sum((x - mean)^2) / N
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"

#ضمن <assert.h>
#ضمن <stdint.h>
#ضمن <string.h>

// Norm kernel parameters structure
بنية ggml_et_norm_params {
    بنية ggml_tensor src0;  // F32 input tensor
    بنية ggml_tensor dst;   // F32 output tensor
    عائم              eps;   // Epsilon parameter for numerical stability
};

صحيح entry_point(بنية ggml_et_norm_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحيح لمب_threads = get_لمب_threads(kernel_env->shire_mask);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;  // Invalid pointer
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * dst  = &params->dst;
    عائم                eps  = params->eps;

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;  // Unsupported type combination
    }

    عائم * src0_data = (عائم *) src0->data;
    عائم * dst_data  = (عائم *) dst->data;

    إذا (!src0_data || !dst_data) {
        أرجع -1;  // Null data pointer
    }


    إذا (eps < 0.0f) {
        أرجع -1;  // Invalid epsilon
    }

    ثابت int64_t ne0 = dst->ne[0];
    ثابت int64_t ne1 = dst->ne[1];
    ثابت int64_t ne2 = dst->ne[2];
    ثابت int64_t ne3 = dst->ne[3];

    ثابت size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ثابت size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    إذا (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        أرجع -1;  // Shape mismatch
    }

    ثابت int32_t total_rows    = (int32_t) (ne1 * ne2 * ne3);
    ثابت صحيح     shire_threads = SOC_MINIONS_PER_SHIRE * لمب_HARTS_PER_MINION;

    إذا (total_rows >= shire_threads) {
        // Row-parallel: each thread processes whole rows
        لكل (int64_t i3 = 0; i3 < ne3; i3++) {
            لكل (int64_t i2 = 0; i2 < ne2; i2++) {
                لكل (int64_t i1 = thread_id; i1 < ne1; i1 += لمب_threads) {
                    ثابت عائم * src_ptr =
                        (ثابت عائم *) ((ثابت حرف *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    عائم * dst_ptr = (عائم *) ((حرف *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

                    // Step 1: sum for mean
                    عائم zero = 0.0f;
                    __asm__ متطاير("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                    لكل (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ متطاير(
                            "flw.ps f11, %[x_vec]\n"
                            "fadd.ps f10, f10, f11\n"
                            :
                            : [x_vec] "m"(*(ثابت عائم (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    عائم sum;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(sum)::"t0", "f1", "f2", "f3", "f4", "f5");

                    ثابت عائم mean = et_fdiv(sum, (عائم) (int32_t) ne0);

                    // Step 2: compute (x - mean) → dst, accumulate variance
                    __asm__ متطاير("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                    لكل (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ متطاير(
                            "flw.ps f11, %[x_vec]\n"
                            "fbc.ps f12, %[mean_ptr]\n"
                            "fsub.ps f13, f11, f12\n"
                            "fsw.ps f13, %[result]\n"
                            "fmadd.ps f10, f13, f13, f10\n"
                            : [result] "=m"(*(عائم (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(ثابت عائم (*)[8]) & src_ptr[i0]), [mean_ptr] "m"(mean)
                            : "f10", "f11", "f12", "f13");
                    }

                    عائم var_sum;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(var_sum)::"t0", "f1", "f2", "f3", "f4", "f5");

                    ثابت عائم variance = et_fdiv(var_sum, (عائم) (int32_t) ne0);
                    ثابت عائم scale    = et_powf(variance + eps, -0.5f);

                    إذا (!(scale > 0.0f)) {
                        أرجع -1;
                    }

                    // Step 3: apply scale to centered values in dst
                    لكل (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ متطاير(
                            "flw.ps f12, %[y_vec]\n"
                            "fbc.ps f13, %[scale_ptr]\n"
                            "fmul.ps f14, f12, f13\n"
                            "fsw.ps f14, %[result]\n"
                            : [result] "=m"(*(عائم (*)[8]) & dst_ptr[i0])
                            : [y_vec] "m"(*(ثابت عائم (*)[8]) & dst_ptr[i0]), [scale_ptr] "m"(scale)
                            : "f12", "f13", "f14");
                    }
                }
            }
        }
    } وإلا {
        // Intra-row: threads within each shire cooperate via L2 SCP.
        // Two reductions needed: sum (for mean), then variance sum.
        صحيح shire_tid       = thread_id % shire_threads;
        صحيح threads_per_row = shire_threads / total_rows;
        صحيح my_row          = shire_tid / threads_per_row;
        صحيح local_tid       = shire_tid % threads_per_row;
        صحيح group_base      = my_row * threads_per_row;

        إذا (my_row >= total_rows) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            // Second barrier for variance exchange
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            أرجع 0;
        }

        int64_t i1 = my_row % ne1;
        int64_t i2 = (my_row / ne1) % ne2;
        int64_t i3 = my_row / (ne1 * ne2);

        ثابت عائم * src_ptr = (ثابت عائم *) ((ثابت حرف *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
        عائم *       dst_ptr = (عائم *) ((حرف *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

        ثابت int32_t elems_per_cl   = 16;
        int32_t       total_cls      = ((int32_t) ne0 + elems_per_cl - 1) / elems_per_cl;
        int32_t       cls_per_thread = (total_cls + threads_per_row - 1) / threads_per_row;
        int32_t       my_start       = local_tid * cls_per_thread * elems_per_cl;
        int32_t       my_end         = my_start + cls_per_thread * elems_per_cl;
        إذا (my_end > (int32_t) ne0) {
            my_end = (int32_t) ne0;
        }
        إذا (my_start >= (int32_t) ne0) {
            my_start = 0;
            my_end   = 0;
        }

        صحيح workers = threads_per_row < total_cls ? threads_per_row : total_cls;

        unsigned long saved_mask;
        __asm__ متطاير("mova.x.m %0" : "=r"(saved_mask));
        __asm__ متطاير("mov.m.x m0, x0, 0xFF");

        // ---- Reduction 1: partial sum for mean ----
        __asm__ متطاير("fbci.pi f10, 0" ::: "f10");
        لكل (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
            __asm__ متطاير(
                "flw.ps f11, %[x_vec]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x_vec] "m"(*(ثابت عائم (*)[8]) & src_ptr[i0])
                : "f10", "f11");
        }

        عائم partial_sum;
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
        متطاير عائم * my_slot = (متطاير عائم *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_sum;
        FENCE;
        evict_to_l2((ثابت فراغ *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // All threads read sum, compute mean
        لكل (صحيح t = 0; t < workers; t++) {
            متطاير عائم * slot = (متطاير عائم *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            evict_to_l2((ثابت فراغ *) slot, 1, 64);
        }
        WAIT_CACHEOPS;

        عائم total_sum = 0.0f;
        لكل (صحيح t = 0; t < workers; t++) {
            متطاير عائم * slot = (متطاير عائم *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            total_sum += *slot;
        }

        ثابت عائم mean = et_fdiv(total_sum, (عائم) (int32_t) ne0);

        // ---- Reduction 2: compute (x - mean) → dst chunk, partial variance ----
        __asm__ متطاير("fbci.pi f10, 0" ::: "f10");

        إذا (my_start < my_end) {
            uint32_t mean_bits;
            __asm__ متطاير("fmv.x.s %0, %1" : "=r"(mean_bits) : "f"(mean));
            __asm__ متطاير("fbcx.ps f15, %[mb]\n" : : [mb] "r"(mean_bits) : "f15");

            لكل (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                __asm__ متطاير(
                    "flw.ps f11, %[x_vec]\n"
                    "fsub.ps f13, f11, f15\n"
                    "fsw.ps f13, %[result]\n"
                    "fmadd.ps f10, f13, f13, f10\n"
                    : [result] "=m"(*(عائم (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(ثابت عائم (*)[8]) & src_ptr[i0])
                    : "f10", "f11", "f13");
            }
        }

        عائم partial_var;
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
        evict_to_l2((ثابت فراغ *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // All threads read variance, compute scale, apply to own chunk
        لكل (صحيح t = 0; t < workers; t++) {
            متطاير عائم * slot = (متطاير عائم *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            evict_to_l2((ثابت فراغ *) slot, 1, 64);
        }
        WAIT_CACHEOPS;

        عائم total_var = 0.0f;
        لكل (صحيح t = 0; t < workers; t++) {
            متطاير عائم * slot = (متطاير عائم *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            total_var += *slot;
        }

        ثابت عائم variance = et_fdiv(total_var, (عائم) (int32_t) ne0);
        ثابت عائم scale    = et_powf(variance + eps, -0.5f);

        إذا (!(scale > 0.0f)) {
            __asm__ متطاير("mova.m.x %0" ::"r"(saved_mask));
            أرجع -1;
        }

        // Apply scale to centered values (already in dst from reduction 2)
        إذا (my_start < my_end) {
            uint32_t scale_bits;
            __asm__ متطاير("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
            __asm__ متطاير("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

            لكل (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                __asm__ متطاير(
                    "flw.ps f12, %[y_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fsw.ps f14, %[result]\n"
                    : [result] "=m"(*(عائم (*)[8]) & dst_ptr[i0])
                    : [y_vec] "m"(*(ثابت عائم (*)[8]) & dst_ptr[i0])
                    : "f12", "f14");
            }
        }

        __asm__ متطاير("mova.m.x %0" ::"r"(saved_mask));
    }

    أرجع 0;
}
