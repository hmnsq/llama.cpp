//******************************************************************************
// Norm F32 Kernel (Layer Normalization)
// y[i] = (x[i] - mean) / sqrt(variance + eps)
// where mean = sum(x) / N, variance = sum((x - mean)^2) / N
//******************************************************************************

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "math_fp.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <assert.h>
#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <string.h>

// Norm kernel parameters structure
সংরচনা ggml_et_norm_params {
    সংরচনা ggml_tensor src0;  // F32 input tensor
    সংরচনা ggml_tensor dst;   // F32 output tensor
    ভিন্ন              eps;   // Epsilon parameter for numerical stability
};

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_norm_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        প্রত্যাগম -1;
    }

    পূর্ণাঙ্ক thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণাঙ্ক অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        প্রত্যাগম 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        প্রত্যাগম -1;  // Invalid pointer
    }

    সংরচনা ggml_tensor * src0 = &params->src0;
    সংরচনা ggml_tensor * dst  = &params->dst;
    ভিন্ন                eps  = params->eps;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        প্রত্যাগম -1;  // Unsupported type combination
    }

    ভিন্ন * src0_data = (ভিন্ন *) src0->data;
    ভিন্ন * dst_data  = (ভিন্ন *) dst->data;

    যদি (!src0_data || !dst_data) {
        প্রত্যাগম -1;  // Null data pointer
    }


    যদি (eps < 0.0f) {
        প্রত্যাগম -1;  // Invalid epsilon
    }

    ধ্রুব int64_t ne0 = dst->ne[0];
    ধ্রুব int64_t ne1 = dst->ne[1];
    ধ্রুব int64_t ne2 = dst->ne[2];
    ধ্রুব int64_t ne3 = dst->ne[3];

    ধ্রুব size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ধ্রুব size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    যদি (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        প্রত্যাগম -1;  // Shape mismatch
    }

    ধ্রুব int32_t total_rows    = (int32_t) (ne1 * ne2 * ne3);
    ধ্রুব পূর্ণাঙ্ক     shire_threads = SOC_MINIONS_PER_SHIRE * অঙ্ক_HARTS_PER_MINION;

    যদি (total_rows >= shire_threads) {
        // Row-parallel: each thread processes whole rows
        প্রতি (int64_t i3 = 0; i3 < ne3; i3++) {
            প্রতি (int64_t i2 = 0; i2 < ne2; i2++) {
                প্রতি (int64_t i1 = thread_id; i1 < ne1; i1 += অঙ্ক_threads) {
                    ধ্রুব ভিন্ন * src_ptr =
                        (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    ভিন্ন * dst_ptr = (ভিন্ন *) ((বর্ণ *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

                    // Step 1: sum for mean
                    ভিন্ন zero = 0.0f;
                    __asm__ চঞ্চল("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                    প্রতি (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ চঞ্চল(
                            "flw.ps f11, %[x_vec]\n"
                            "fadd.ps f10, f10, f11\n"
                            :
                            : [x_vec] "m"(*(ধ্রুব ভিন্ন (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    ভিন্ন sum;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(sum)::"t0", "f1", "f2", "f3", "f4", "f5");

                    ধ্রুব ভিন্ন mean = et_fdiv(sum, (ভিন্ন) (int32_t) ne0);

                    // Step 2: compute (x - mean) → dst, accumulate variance
                    __asm__ চঞ্চল("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                    প্রতি (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ চঞ্চল(
                            "flw.ps f11, %[x_vec]\n"
                            "fbc.ps f12, %[mean_ptr]\n"
                            "fsub.ps f13, f11, f12\n"
                            "fsw.ps f13, %[result]\n"
                            "fmadd.ps f10, f13, f13, f10\n"
                            : [result] "=m"(*(ভিন্ন (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(ধ্রুব ভিন্ন (*)[8]) & src_ptr[i0]), [mean_ptr] "m"(mean)
                            : "f10", "f11", "f12", "f13");
                    }

                    ভিন্ন var_sum;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(var_sum)::"t0", "f1", "f2", "f3", "f4", "f5");

                    ধ্রুব ভিন্ন variance = et_fdiv(var_sum, (ভিন্ন) (int32_t) ne0);
                    ধ্রুব ভিন্ন scale    = et_powf(variance + eps, -0.5f);

                    যদি (!(scale > 0.0f)) {
                        প্রত্যাগম -1;
                    }

                    // Step 3: apply scale to centered values in dst
                    প্রতি (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ চঞ্চল(
                            "flw.ps f12, %[y_vec]\n"
                            "fbc.ps f13, %[scale_ptr]\n"
                            "fmul.ps f14, f12, f13\n"
                            "fsw.ps f14, %[result]\n"
                            : [result] "=m"(*(ভিন্ন (*)[8]) & dst_ptr[i0])
                            : [y_vec] "m"(*(ধ্রুব ভিন্ন (*)[8]) & dst_ptr[i0]), [scale_ptr] "m"(scale)
                            : "f12", "f13", "f14");
                    }
                }
            }
        }
    } অন্যথা {
        // Intra-row: threads within each shire cooperate via L2 SCP.
        // Two reductions needed: sum (for mean), then variance sum.
        পূর্ণাঙ্ক shire_tid       = thread_id % shire_threads;
        পূর্ণাঙ্ক threads_per_row = shire_threads / total_rows;
        পূর্ণাঙ্ক my_row          = shire_tid / threads_per_row;
        পূর্ণাঙ্ক local_tid       = shire_tid % threads_per_row;
        পূর্ণাঙ্ক group_base      = my_row * threads_per_row;

        যদি (my_row >= total_rows) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            // Second barrier for variance exchange
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            প্রত্যাগম 0;
        }

        int64_t i1 = my_row % ne1;
        int64_t i2 = (my_row / ne1) % ne2;
        int64_t i3 = my_row / (ne1 * ne2);

        ধ্রুব ভিন্ন * src_ptr = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
        ভিন্ন *       dst_ptr = (ভিন্ন *) ((বর্ণ *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

        ধ্রুব int32_t elems_per_cl   = 16;
        int32_t       total_cls      = ((int32_t) ne0 + elems_per_cl - 1) / elems_per_cl;
        int32_t       cls_per_thread = (total_cls + threads_per_row - 1) / threads_per_row;
        int32_t       my_start       = local_tid * cls_per_thread * elems_per_cl;
        int32_t       my_end         = my_start + cls_per_thread * elems_per_cl;
        যদি (my_end > (int32_t) ne0) {
            my_end = (int32_t) ne0;
        }
        যদি (my_start >= (int32_t) ne0) {
            my_start = 0;
            my_end   = 0;
        }

        পূর্ণাঙ্ক workers = threads_per_row < total_cls ? threads_per_row : total_cls;

        unsigned long saved_mask;
        __asm__ চঞ্চল("mova.x.m %0" : "=r"(saved_mask));
        __asm__ চঞ্চল("mov.m.x m0, x0, 0xFF");

        // ---- Reduction 1: partial sum for mean ----
        __asm__ চঞ্চল("fbci.pi f10, 0" ::: "f10");
        প্রতি (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
            __asm__ চঞ্চল(
                "flw.ps f11, %[x_vec]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x_vec] "m"(*(ধ্রুব ভিন্ন (*)[8]) & src_ptr[i0])
                : "f10", "f11");
        }

        ভিন্ন partial_sum;
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
        চঞ্চল ভিন্ন * my_slot = (চঞ্চল ভিন্ন *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_sum;
        FENCE;
        evict_to_l2((ধ্রুব শূন্য *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // All threads read sum, compute mean
        প্রতি (পূর্ণাঙ্ক t = 0; t < workers; t++) {
            চঞ্চল ভিন্ন * slot = (চঞ্চল ভিন্ন *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            evict_to_l2((ধ্রুব শূন্য *) slot, 1, 64);
        }
        WAIT_CACHEOPS;

        ভিন্ন total_sum = 0.0f;
        প্রতি (পূর্ণাঙ্ক t = 0; t < workers; t++) {
            চঞ্চল ভিন্ন * slot = (চঞ্চল ভিন্ন *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            total_sum += *slot;
        }

        ধ্রুব ভিন্ন mean = et_fdiv(total_sum, (ভিন্ন) (int32_t) ne0);

        // ---- Reduction 2: compute (x - mean) → dst chunk, partial variance ----
        __asm__ চঞ্চল("fbci.pi f10, 0" ::: "f10");

        যদি (my_start < my_end) {
            uint32_t mean_bits;
            __asm__ চঞ্চল("fmv.x.s %0, %1" : "=r"(mean_bits) : "f"(mean));
            __asm__ চঞ্চল("fbcx.ps f15, %[mb]\n" : : [mb] "r"(mean_bits) : "f15");

            প্রতি (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                __asm__ চঞ্চল(
                    "flw.ps f11, %[x_vec]\n"
                    "fsub.ps f13, f11, f15\n"
                    "fsw.ps f13, %[result]\n"
                    "fmadd.ps f10, f13, f13, f10\n"
                    : [result] "=m"(*(ভিন্ন (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(ধ্রুব ভিন্ন (*)[8]) & src_ptr[i0])
                    : "f10", "f11", "f13");
            }
        }

        ভিন্ন partial_var;
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
        evict_to_l2((ধ্রুব শূন্য *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // All threads read variance, compute scale, apply to own chunk
        প্রতি (পূর্ণাঙ্ক t = 0; t < workers; t++) {
            চঞ্চল ভিন্ন * slot = (চঞ্চল ভিন্ন *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            evict_to_l2((ধ্রুব শূন্য *) slot, 1, 64);
        }
        WAIT_CACHEOPS;

        ভিন্ন total_var = 0.0f;
        প্রতি (পূর্ণাঙ্ক t = 0; t < workers; t++) {
            চঞ্চল ভিন্ন * slot = (চঞ্চল ভিন্ন *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            total_var += *slot;
        }

        ধ্রুব ভিন্ন variance = et_fdiv(total_var, (ভিন্ন) (int32_t) ne0);
        ধ্রুব ভিন্ন scale    = et_powf(variance + eps, -0.5f);

        যদি (!(scale > 0.0f)) {
            __asm__ চঞ্চল("mova.m.x %0" ::"r"(saved_mask));
            প্রত্যাগম -1;
        }

        // Apply scale to centered values (already in dst from reduction 2)
        যদি (my_start < my_end) {
            uint32_t scale_bits;
            __asm__ চঞ্চল("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
            __asm__ চঞ্চল("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

            প্রতি (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                __asm__ চঞ্চল(
                    "flw.ps f12, %[y_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fsw.ps f14, %[result]\n"
                    : [result] "=m"(*(ভিন্ন (*)[8]) & dst_ptr[i0])
                    : [y_vec] "m"(*(ধ্রুব ভিন্ন (*)[8]) & dst_ptr[i0])
                    : "f12", "f14");
            }
        }

        __asm__ চঞ্চল("mova.m.x %0" ::"r"(saved_mask));
    }

    প্রত্যাগম 0;
}
