//******************************************************************************
// L2 Norm F32 Kernel (L2 Normalization)
// y[i] = x[i] / max(sqrt(sum(x^2)), eps)
//******************************************************************************

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "math_fp.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <assert.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <string.h>

// L2 Norm kernel parameters structure
संरचना ggml_et_l2_norm_params {
    संरचना ggml_tensor src0;  // F32 input tensor
    संरचना ggml_tensor dst;   // F32 output tensor
    भिन्न              eps;   // Epsilon parameter for numerical stability
};

पूर्णाङ्क entry_point(संरचना ggml_et_l2_norm_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जइ (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अङ्कः_threads = get_अङ्कः_threads(kernel_env->shire_mask);

    जइ (thread_id < 0) {
        प्रत्यागम 0;
    }

    जइ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;  // Invalid pointer
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;
    भिन्न                eps  = params->eps;

    जइ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;  // Unsupported type combination
    }

    भिन्न * src0_data = (भिन्न *) src0->data;
    भिन्न * dst_data  = (भिन्न *) dst->data;

    जइ (!src0_data || !dst_data) {
        प्रत्यागम -1;  // Null data pointer
    }

    जइ (eps < 0.0f) {
        प्रत्यागम -1;  // Invalid epsilon
    }

    ध्रुव int64_t ne0 = dst->ne[0];
    ध्रुव int64_t ne1 = dst->ne[1];
    ध्रुव int64_t ne2 = dst->ne[2];
    ध्रुव int64_t ne3 = dst->ne[3];

    ध्रुव size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ध्रुव size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    (शून्य) nb0;
    (शून्य) nb00;

    जइ (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        प्रत्यागम -1;  // Shape mismatch
    }

    ध्रुव int32_t total_rows    = (int32_t) (ne1 * ne2 * ne3);
    ध्रुव पूर्णाङ्क     shire_threads = SOC_MINIONS_PER_SHIRE * अङ्कः_HARTS_PER_MINION;

    जइ (total_rows >= shire_threads) {
        // Row-parallel: each thread processes whole rows
        प्रति (int64_t i3 = 0; i3 < ne3; i3++) {
            प्रति (int64_t i2 = 0; i2 < ne2; i2++) {
                प्रति (int64_t i1 = thread_id; i1 < ne1; i1 += अङ्कः_threads) {
                    ध्रुव भिन्न * src_ptr =
                        (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    भिन्न * dst_ptr = (भिन्न *) ((वर्ण *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

                    भिन्न zero = 0.0f;
                    __asm__ चञ्चल("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                    प्रति (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ चञ्चल(
                            "flw.ps f11, %[x_vec]\n"
                            "fmadd.ps f10, f11, f11, f10\n"
                            :
                            : [x_vec] "m"(*(ध्रुव भिन्न (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    भिन्न sum_sq;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(sum_sq)::"t0", "f1", "f2", "f3", "f4", "f5");

                    भिन्न l2_norm = et_powf(sum_sq, 0.5f);
                    जइ (l2_norm < eps) {
                        l2_norm = eps;
                    }
                    ध्रुव भिन्न scale = et_fdiv(1.0f, l2_norm);

                    प्रति (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ चञ्चल(
                            "flw.ps f11, %[x_vec]\n"
                            "fbc.ps f12, %[scale_ptr]\n"
                            "fmul.ps f13, f11, f12\n"
                            "fsw.ps f13, %[result]\n"
                            : [result] "=m"(*(भिन्न (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(ध्रुव भिन्न (*)[8]) & src_ptr[i0]), [scale_ptr] "m"(scale)
                            : "f11", "f12", "f13");
                    }
                }
            }
        }
    } अण्णहा {
        // Intra-row: threads within each shire cooperate via L2 SCP
        पूर्णाङ्क shire_tid       = thread_id % shire_threads;
        पूर्णाङ्क threads_per_row = shire_threads / total_rows;
        पूर्णाङ्क my_row          = shire_tid / threads_per_row;
        पूर्णाङ्क local_tid       = shire_tid % threads_per_row;
        पूर्णाङ्क group_base      = my_row * threads_per_row;

        जइ (my_row >= total_rows) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            प्रत्यागम 0;
        }

        int64_t i1 = my_row % ne1;
        int64_t i2 = (my_row / ne1) % ne2;
        int64_t i3 = my_row / (ne1 * ne2);

        ध्रुव भिन्न * src_ptr = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
        भिन्न *       dst_ptr = (भिन्न *) ((वर्ण *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);

        ध्रुव int32_t elems_per_cl   = 16;
        int32_t       total_cls      = ((int32_t) ne0 + elems_per_cl - 1) / elems_per_cl;
        int32_t       cls_per_thread = (total_cls + threads_per_row - 1) / threads_per_row;
        int32_t       my_start       = local_tid * cls_per_thread * elems_per_cl;
        int32_t       my_end         = my_start + cls_per_thread * elems_per_cl;
        जइ (my_end > (int32_t) ne0) {
            my_end = (int32_t) ne0;
        }
        जइ (my_start >= (int32_t) ne0) {
            my_start = 0;
            my_end   = 0;
        }

        unsigned long saved_mask;
        __asm__ चञ्चल("mova.x.m %0" : "=r"(saved_mask));
        __asm__ चञ्चल("mov.m.x m0, x0, 0xFF");

        // Phase 1: partial sum of squares
        __asm__ चञ्चल("fbci.pi f10, 0" ::: "f10");
        प्रति (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
            __asm__ चञ्चल(
                "flw.ps f11, %[x_vec]\n"
                "fmadd.ps f10, f11, f11, f10\n"
                :
                : [x_vec] "m"(*(ध्रुव भिन्न (*)[8]) & src_ptr[i0])
                : "f10", "f11");
        }

        भिन्न partial_sum;
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
        चञ्चल भिन्न * my_slot = (चञ्चल भिन्न *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_sum;
        FENCE;
        evict_to_l2((ध्रुव शून्य *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // Phase 3: all threads reduce + apply scale to own chunk
        पूर्णाङ्क workers = threads_per_row < total_cls ? threads_per_row : total_cls;

        प्रति (पूर्णाङ्क t = 0; t < workers; t++) {
            चञ्चल भिन्न * slot = (चञ्चल भिन्न *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            evict_to_l2((ध्रुव शून्य *) slot, 1, 64);
        }
        WAIT_CACHEOPS;

        भिन्न total_sum_sq = 0.0f;
        प्रति (पूर्णाङ्क t = 0; t < workers; t++) {
            चञ्चल भिन्न * slot = (चञ्चल भिन्न *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            total_sum_sq += *slot;
        }

        भिन्न l2_norm = et_powf(total_sum_sq, 0.5f);
        जइ (l2_norm < eps) {
            l2_norm = eps;
        }
        ध्रुव भिन्न scale = et_fdiv(1.0f, l2_norm);

        जइ (my_start < my_end) {
            uint32_t scale_bits;
            __asm__ चञ्चल("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
            __asm__ चञ्चल("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

            प्रति (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                __asm__ चञ्चल(
                    "flw.ps f12, %[x_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fsw.ps f14, %[result]\n"
                    : [result] "=m"(*(भिन्न (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(ध्रुव भिन्न (*)[8]) & src_ptr[i0])
                    : "f12", "f14");
            }
        }

        __asm__ चञ्चल("mova.m.x %0" ::"r"(saved_mask));
    }

    प्रत्यागम 0;
}
