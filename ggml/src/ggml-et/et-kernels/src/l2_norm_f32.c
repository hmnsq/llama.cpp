//******************************************************************************
// L2 Norm F32 Kernel (L2 Normalization)
// y[i] = x[i] / max(sqrt(sum(x^2)), eps)
//******************************************************************************

#include "ggml_tensor.h"
#include "math_fp.h"
#include "platform.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

// L2 Norm kernel parameters structure
struct ggml_et_l2_norm_params {
    struct ggml_tensor src0;  // F32 input tensor
    struct ggml_tensor dst;   // F32 output tensor
    float              eps;   // Epsilon parameter for numerical stability
};

цео entry_point(struct ggml_et_l2_norm_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ако (!kernel_env) {
        врати -1;
    }

    цео thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    цео num_threads = get_num_threads(kernel_env->shire_mask);

    ако (thread_id < 0) {
        врати 0;
    }

    ако (params == 0 || ((uint64_t) params & 0x7) != 0) {
        врати -1;  // Invalid pointer
    }

    struct ggml_tensor * src0 = &params->src0;
    struct ggml_tensor * dst  = &params->dst;
    float                eps  = params->eps;

    ако (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        врати -1;  // Unsupported type combination
    }

    float * src0_data = (float *) src0->data;
    float * dst_data  = (float *) dst->data;

    ако (!src0_data || !dst_data) {
        врати -1;  // Null data pointer
    }

    ако (eps < 0.0f) {
        врати -1;  // Invalid epsilon
    }

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne3 = dst->ne[3];

    const size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    const size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    (void) nb0;
    (void) nb00;

    ако (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        врати -1;  // Shape mismatch
    }

    const int32_t total_rows    = (int32_t) (ne1 * ne2 * ne3);
    const цео     shire_threads = SOC_MINIONS_PER_SHIRE * NUM_HARTS_PER_MINION;

    ако (total_rows >= shire_threads) {
        // Row-parallel: each thread processes whole rows
        за (int64_t i3 = 0; i3 < ne3; i3++) {
            за (int64_t i2 = 0; i2 < ne2; i2++) {
                за (int64_t i1 = thread_id; i1 < ne1; i1 += num_threads) {
                    const float * src_ptr =
                        (const float *) ((const char *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    float * dst_ptr = (float *) ((char *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

                    float zero = 0.0f;
                    __asm__ volatile("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                    за (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ volatile(
                            "flw.ps f11, %[x_vec]\n"
                            "fmadd.ps f10, f11, f11, f10\n"
                            :
                            : [x_vec] "m"(*(const float (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    float sum_sq;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(sum_sq)::"t0", "f1", "f2", "f3", "f4", "f5");

                    float l2_norm = et_powf(sum_sq, 0.5f);
                    ако (l2_norm < eps) {
                        l2_norm = eps;
                    }
                    const float scale = et_fdiv(1.0f, l2_norm);

                    за (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ volatile(
                            "flw.ps f11, %[x_vec]\n"
                            "fbc.ps f12, %[scale_ptr]\n"
                            "fmul.ps f13, f11, f12\n"
                            "fsw.ps f13, %[result]\n"
                            : [result] "=m"(*(float (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(const float (*)[8]) & src_ptr[i0]), [scale_ptr] "m"(scale)
                            : "f11", "f12", "f13");
                    }
                }
            }
        }
    } инако {
        // Intra-row: threads within each shire cooperate via L2 SCP
        цео shire_tid       = thread_id % shire_threads;
        цео threads_per_row = shire_threads / total_rows;
        цео my_row          = shire_tid / threads_per_row;
        цео local_tid       = shire_tid % threads_per_row;
        цео group_base      = my_row * threads_per_row;

        ако (my_row >= total_rows) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            врати 0;
        }

        int64_t i1 = my_row % ne1;
        int64_t i2 = (my_row / ne1) % ne2;
        int64_t i3 = my_row / (ne1 * ne2);

        const float * src_ptr = (const float *) ((const char *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
        float *       dst_ptr = (float *) ((char *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

        const int32_t elems_per_cl   = 16;
        int32_t       total_cls      = ((int32_t) ne0 + elems_per_cl - 1) / elems_per_cl;
        int32_t       cls_per_thread = (total_cls + threads_per_row - 1) / threads_per_row;
        int32_t       my_start       = local_tid * cls_per_thread * elems_per_cl;
        int32_t       my_end         = my_start + cls_per_thread * elems_per_cl;
        ако (my_end > (int32_t) ne0) {
            my_end = (int32_t) ne0;
        }
        ако (my_start >= (int32_t) ne0) {
            my_start = 0;
            my_end   = 0;
        }

        unsigned long saved_mask;
        __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
        __asm__ volatile("mov.m.x m0, x0, 0xFF");

        // Phase 1: partial sum of squares
        __asm__ volatile("fbci.pi f10, 0" ::: "f10");
        за (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
            __asm__ volatile(
                "flw.ps f11, %[x_vec]\n"
                "fmadd.ps f10, f11, f11, f10\n"
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

        // Phase 2: L2SCP exchange
        volatile float * my_slot = (volatile float *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_sum;
        FENCE;
        evict_to_l2((const void *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // Phase 3: all threads reduce + apply scale to own chunk
        цео workers = threads_per_row < total_cls ? threads_per_row : total_cls;

        за (цео t = 0; t < workers; t++) {
            volatile float * slot = (volatile float *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            evict_to_l2((const void *) slot, 1, 64);
        }
        WAIT_CACHEOPS;

        float total_sum_sq = 0.0f;
        за (цео t = 0; t < workers; t++) {
            volatile float * slot = (volatile float *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            total_sum_sq += *slot;
        }

        float l2_norm = et_powf(total_sum_sq, 0.5f);
        ако (l2_norm < eps) {
            l2_norm = eps;
        }
        const float scale = et_fdiv(1.0f, l2_norm);

        ако (my_start < my_end) {
            uint32_t scale_bits;
            __asm__ volatile("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
            __asm__ volatile("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

            за (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                __asm__ volatile(
                    "flw.ps f12, %[x_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fsw.ps f14, %[result]\n"
                    : [result] "=m"(*(float (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(const float (*)[8]) & src_ptr[i0])
                    : "f12", "f14");
            }
        }

        __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
    }

    врати 0;
}
