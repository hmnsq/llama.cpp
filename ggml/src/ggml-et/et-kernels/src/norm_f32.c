//******************************************************************************
// Norm F32 Kernel (Layer Normalization)
// y[i] = (x[i] - mean) / sqrt(variance + eps)
// where mean = sum(x) / N, variance = sum((x - mean)^2) / N
//******************************************************************************

#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

// Norm kernel parameters structure
sa_nracanaa ggml_et_norm_params {
    sa_nracanaa ggml_tensor src0;  // F32 input tensor
    sa_nracanaa ggml_tensor dst;   // F32 output tensor
    float              eps;   // Epsilon parameter for numerical stability
};

ကိန်းပြည့် entry_point(sa_nracanaa ggml_et_norm_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    အကယ်၍ (!kernel_env) {
        ပြန်ပေး -1;
    }

    ကိန်းပြည့် thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ကိန်းပြည့် num_threads = get_num_threads(kernel_env->shire_mask);

    အကယ်၍ (thread_id < 0) {
        ပြန်ပေး 0;
    }

    အကယ်၍ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ပြန်ပေး -1;  // Invalid pointer
    }

    sa_nracanaa ggml_tensor * src0 = &params->src0;
    sa_nracanaa ggml_tensor * dst  = &params->dst;
    float                eps  = params->eps;

    အကယ်၍ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ပြန်ပေး -1;  // Unsupported type combination
    }

    float * src0_data = (float *) src0->data;
    float * dst_data  = (float *) dst->data;

    အကယ်၍ (!src0_data || !dst_data) {
        ပြန်ပေး -1;  // Null data pointer
    }


    အကယ်၍ (eps < 0.0f) {
        ပြန်ပေး -1;  // Invalid epsilon
    }

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    const size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    const size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    အကယ်၍ (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        ပြန်ပေး -1;  // Shape mismatch
    }

    const int32_t total_rows    = (int32_t) (ne1 * ne2 * ne3);
    const ကိန်းပြည့်     shire_threads = SOC_MINIONS_PER_SHIRE * NUM_HARTS_PER_MINION;

    အကယ်၍ (total_rows >= shire_threads) {
        // Row-parallel: each thread processes whole rows
        အတွက် (int64_t i3 = 0; i3 < ne3; i3++) {
            အတွက် (int64_t i2 = 0; i2 < ne2; i2++) {
                အတွက် (int64_t i1 = thread_id; i1 < ne1; i1 += num_threads) {
                    const float * src_ptr =
                        (const float *) ((const var_na *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    float * dst_ptr = (float *) ((var_na *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

                    // Step 1: sum for mean
                    float zero = 0.0f;
                    __asm__ ca_ncala("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                    အတွက် (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ ca_ncala(
                            "flw.ps f11, %[x_vec]\n"
                            "fadd.ps f10, f10, f11\n"
                            :
                            : [x_vec] "m"(*(const float (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    float sum;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(sum)::"t0", "f1", "f2", "f3", "f4", "f5");

                    const float mean = et_fdiv(sum, (float) (int32_t) ne0);

                    // Step 2: compute (x - mean) → dst, accumulate variance
                    __asm__ ca_ncala("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                    အတွက် (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ ca_ncala(
                            "flw.ps f11, %[x_vec]\n"
                            "fbc.ps f12, %[mean_ptr]\n"
                            "fsub.ps f13, f11, f12\n"
                            "fsw.ps f13, %[result]\n"
                            "fmadd.ps f10, f13, f13, f10\n"
                            : [result] "=m"(*(float (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(const float (*)[8]) & src_ptr[i0]), [mean_ptr] "m"(mean)
                            : "f10", "f11", "f12", "f13");
                    }

                    float var_sum;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(var_sum)::"t0", "f1", "f2", "f3", "f4", "f5");

                    const float variance = et_fdiv(var_sum, (float) (int32_t) ne0);
                    const float scale    = et_powf(variance + eps, -0.5f);

                    အကယ်၍ (!(scale > 0.0f)) {
                        ပြန်ပေး -1;
                    }

                    // Step 3: apply scale to centered values in dst
                    အတွက် (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ ca_ncala(
                            "flw.ps f12, %[y_vec]\n"
                            "fbc.ps f13, %[scale_ptr]\n"
                            "fmul.ps f14, f12, f13\n"
                            "fsw.ps f14, %[result]\n"
                            : [result] "=m"(*(float (*)[8]) & dst_ptr[i0])
                            : [y_vec] "m"(*(const float (*)[8]) & dst_ptr[i0]), [scale_ptr] "m"(scale)
                            : "f12", "f13", "f14");
                    }
                }
            }
        }
    } မဟုတ်ပါက {
        // Intra-row: threads within each shire cooperate via L2 SCP.
        // Two reductions needed: sum (for mean), then variance sum.
        ကိန်းပြည့် shire_tid       = thread_id % shire_threads;
        ကိန်းပြည့် threads_per_row = shire_threads / total_rows;
        ကိန်းပြည့် my_row          = shire_tid / threads_per_row;
        ကိန်းပြည့် local_tid       = shire_tid % threads_per_row;
        ကိန်းပြည့် group_base      = my_row * threads_per_row;

        အကယ်၍ (my_row >= total_rows) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            // Second barrier for variance exchange
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            ပြန်ပေး 0;
        }

        int64_t i1 = my_row % ne1;
        int64_t i2 = (my_row / ne1) % ne2;
        int64_t i3 = my_row / (ne1 * ne2);

        const float * src_ptr = (const float *) ((const var_na *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
        float *       dst_ptr = (float *) ((var_na *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

        const int32_t elems_per_cl   = 16;
        int32_t       total_cls      = ((int32_t) ne0 + elems_per_cl - 1) / elems_per_cl;
        int32_t       cls_per_thread = (total_cls + threads_per_row - 1) / threads_per_row;
        int32_t       my_start       = local_tid * cls_per_thread * elems_per_cl;
        int32_t       my_end         = my_start + cls_per_thread * elems_per_cl;
        အကယ်၍ (my_end > (int32_t) ne0) {
            my_end = (int32_t) ne0;
        }
        အကယ်၍ (my_start >= (int32_t) ne0) {
            my_start = 0;
            my_end   = 0;
        }

        ကိန်းပြည့် workers = threads_per_row < total_cls ? threads_per_row : total_cls;

        unsigned long saved_mask;
        __asm__ ca_ncala("mova.x.m %0" : "=r"(saved_mask));
        __asm__ ca_ncala("mov.m.x m0, x0, 0xFF");

        // ---- Reduction 1: partial sum for mean ----
        __asm__ ca_ncala("fbci.pi f10, 0" ::: "f10");
        အတွက် (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
            __asm__ ca_ncala(
                "flw.ps f11, %[x_vec]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x_vec] "m"(*(const float (*)[8]) & src_ptr[i0])
                : "f10", "f11");
        }

        float partial_sum;
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
        ca_ncala float * my_slot = (ca_ncala float *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_sum;
        FENCE;
        evict_to_l2((const void *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // All threads read sum, compute mean
        အတွက် (ကိန်းပြည့် t = 0; t < workers; t++) {
            ca_ncala float * slot = (ca_ncala float *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            evict_to_l2((const void *) slot, 1, 64);
        }
        WAIT_CACHEOPS;

        float total_sum = 0.0f;
        အတွက် (ကိန်းပြည့် t = 0; t < workers; t++) {
            ca_ncala float * slot = (ca_ncala float *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            total_sum += *slot;
        }

        const float mean = et_fdiv(total_sum, (float) (int32_t) ne0);

        // ---- Reduction 2: compute (x - mean) → dst chunk, partial variance ----
        __asm__ ca_ncala("fbci.pi f10, 0" ::: "f10");

        အကယ်၍ (my_start < my_end) {
            uint32_t mean_bits;
            __asm__ ca_ncala("fmv.x.s %0, %1" : "=r"(mean_bits) : "f"(mean));
            __asm__ ca_ncala("fbcx.ps f15, %[mb]\n" : : [mb] "r"(mean_bits) : "f15");

            အတွက် (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                __asm__ ca_ncala(
                    "flw.ps f11, %[x_vec]\n"
                    "fsub.ps f13, f11, f15\n"
                    "fsw.ps f13, %[result]\n"
                    "fmadd.ps f10, f13, f13, f10\n"
                    : [result] "=m"(*(float (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(const float (*)[8]) & src_ptr[i0])
                    : "f10", "f11", "f13");
            }
        }

        float partial_var;
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
        evict_to_l2((const void *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // All threads read variance, compute scale, apply to own chunk
        အတွက် (ကိန်းပြည့် t = 0; t < workers; t++) {
            ca_ncala float * slot = (ca_ncala float *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            evict_to_l2((const void *) slot, 1, 64);
        }
        WAIT_CACHEOPS;

        float total_var = 0.0f;
        အတွက် (ကိန်းပြည့် t = 0; t < workers; t++) {
            ca_ncala float * slot = (ca_ncala float *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            total_var += *slot;
        }

        const float variance = et_fdiv(total_var, (float) (int32_t) ne0);
        const float scale    = et_powf(variance + eps, -0.5f);

        အကယ်၍ (!(scale > 0.0f)) {
            __asm__ ca_ncala("mova.m.x %0" ::"r"(saved_mask));
            ပြန်ပေး -1;
        }

        // Apply scale to centered values (already in dst from reduction 2)
        အကယ်၍ (my_start < my_end) {
            uint32_t scale_bits;
            __asm__ ca_ncala("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
            __asm__ ca_ncala("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

            အတွက် (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                __asm__ ca_ncala(
                    "flw.ps f12, %[y_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fsw.ps f14, %[result]\n"
                    : [result] "=m"(*(float (*)[8]) & dst_ptr[i0])
                    : [y_vec] "m"(*(const float (*)[8]) & dst_ptr[i0])
                    : "f12", "f14");
            }
        }

        __asm__ ca_ncala("mova.m.x %0" ::"r"(saved_mask));
    }

    ပြန်ပေး 0;
}
