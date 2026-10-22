//******************************************************************************
// Norm F32 Kernel (Layer Normalization)
// y[i] = (x[i] - mean) / sqrt(variance + eps)
// where mean = sum(x) / N, variance = sum((x - mean)^2) / N
//******************************************************************************

#समावेश "ggml_tensor.h"
#समावेश "math_fp.h"
#समावेश "platform.h"

#समावेश <assert.h>
#समावेश <stdint.h>
#समावेश <string.h>

// Norm kernel parameters structure
काष्ठा ggml_et_norm_params {
    काष्ठा ggml_tensor src0;  // F32 input tensor
    काष्ठा ggml_tensor dst;   // F32 output tensor
    भग्न              eps;   // Epsilon parameter for numerical stability
};

पूर्णांक entry_point(काष्ठा ggml_et_norm_params * params, व्योम * env) {
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
    काष्ठा ggml_tensor * dst  = &params->dst;
    भग्न                eps  = params->eps;

    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस -1;  // Unsupported type combination
    }

    भग्न * src0_तथ्य = (भग्न *) src0->तथ्य;
    भग्न * dst_तथ्य  = (भग्न *) dst->तथ्य;

    अगर (!src0_तथ्य || !dst_तथ्य) {
        वापस -1;  // Null data pointer
    }


    अगर (eps < 0.0f) {
        वापस -1;  // Invalid epsilon
    }

    स्थिर int64_t ne0 = dst->ne[0];
    स्थिर int64_t ne1 = dst->ne[1];
    स्थिर int64_t ne2 = dst->ne[2];
    स्थिर int64_t ne3 = dst->ne[3];

    स्थिर माप_प्रकार nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    स्थिर माप_प्रकार nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    अगर (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        वापस -1;  // Shape mismatch
    }

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

                    // Step 1: sum for mean
                    भग्न zero = 0.0f;
                    __asm__ अस्थिर("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                    क्रम (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ अस्थिर(
                            "flw.ps f11, %[x_vec]\n"
                            "fadd.ps f10, f10, f11\n"
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

                    स्थिर भग्न mean = et_fdiv(योग_2, (भग्न) (int32_t) ne0);

                    // Step 2: compute (x - mean) → dst, accumulate variance
                    __asm__ अस्थिर("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                    क्रम (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ अस्थिर(
                            "flw.ps f11, %[x_vec]\n"
                            "fbc.ps f12, %[mean_ptr]\n"
                            "fsub.ps f13, f11, f12\n"
                            "fsw.ps f13, %[result]\n"
                            "fmadd.ps f10, f13, f13, f10\n"
                            : [परिणाम] "=m"(*(भग्न (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(स्थिर भग्न (*)[8]) & src_ptr[i0]), [mean_ptr] "m"(mean)
                            : "f10", "f11", "f12", "f13");
                    }

                    भग्न var_योग;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(var_योग)::"t0", "f1", "f2", "f3", "f4", "f5");

                    स्थिर भग्न variance = et_fdiv(var_योग, (भग्न) (int32_t) ne0);
                    स्थिर भग्न scale    = et_powf(variance + eps, -0.5f);

                    अगर (!(scale > 0.0f)) {
                        वापस -1;
                    }

                    // Step 3: apply scale to centered values in dst
                    क्रम (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ अस्थिर(
                            "flw.ps f12, %[y_vec]\n"
                            "fbc.ps f13, %[scale_ptr]\n"
                            "fmul.ps f14, f12, f13\n"
                            "fsw.ps f14, %[result]\n"
                            : [परिणाम] "=m"(*(भग्न (*)[8]) & dst_ptr[i0])
                            : [y_vec] "m"(*(स्थिर भग्न (*)[8]) & dst_ptr[i0]), [scale_ptr] "m"(scale)
                            : "f12", "f13", "f14");
                    }
                }
            }
        }
    } अन्यथा {
        // Intra-row: threads within each shire cooperate via L2 SCP.
        // Two reductions needed: sum (for mean), then variance sum.
        पूर्णांक shire_tid       = thread_id % shire_threads;
        पूर्णांक threads_per_row = shire_threads / योग_rows;
        पूर्णांक my_row          = shire_tid / threads_per_row;
        पूर्णांक local_tid       = shire_tid % threads_per_row;
        पूर्णांक group_base      = my_row * threads_per_row;

        अगर (my_row >= योग_rows) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            // Second barrier for variance exchange
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            वापस 0;
        }

        int64_t i1 = my_row % ne1;
        int64_t i2 = (my_row / ne1) % ne2;
        int64_t i3 = my_row / (ne1 * ne2);

        स्थिर भग्न * src_ptr = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i3 * nb03 + i2 * nb02 + i1 * nb01);
        भग्न *       dst_ptr = (भग्न *) ((अक्षर *) dst_तथ्य + i3 * nb3 + i2 * nb2 + i1 * nb1);

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

        पूर्णांक workers = threads_per_row < योग_cls ? threads_per_row : योग_cls;

        अचिन्हित दीर्घ saved_mask;
        __asm__ अस्थिर("mova.x.m %0" : "=r"(saved_mask));
        __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");

        // ---- Reduction 1: partial sum for mean ----
        __asm__ अस्थिर("fbci.pi f10, 0" ::: "f10");
        क्रम (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
            __asm__ अस्थिर(
                "flw.ps f11, %[x_vec]\n"
                "fadd.ps f10, f10, f11\n"
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

        // L2SCP exchange for sum
        अस्थिर भग्न * my_slot = (अस्थिर भग्न *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_slot                 = partial_योग;
        FENCE;
        evict_to_l2((स्थिर व्योम *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // All threads read sum, compute mean
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

        स्थिर भग्न mean = et_fdiv(योग_योग, (भग्न) (int32_t) ne0);

        // ---- Reduction 2: compute (x - mean) → dst chunk, partial variance ----
        __asm__ अस्थिर("fbci.pi f10, 0" ::: "f10");

        अगर (my_start < my_end) {
            uint32_t mean_bits;
            __asm__ अस्थिर("fmv.x.s %0, %1" : "=r"(mean_bits) : "f"(mean));
            __asm__ अस्थिर("fbcx.ps f15, %[mb]\n" : : [mb] "r"(mean_bits) : "f15");

            क्रम (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                __asm__ अस्थिर(
                    "flw.ps f11, %[x_vec]\n"
                    "fsub.ps f13, f11, f15\n"
                    "fsw.ps f13, %[result]\n"
                    "fmadd.ps f10, f13, f13, f10\n"
                    : [परिणाम] "=m"(*(भग्न (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(स्थिर भग्न (*)[8]) & src_ptr[i0])
                    : "f10", "f11", "f13");
            }
        }

        भग्न partial_var;
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
        evict_to_l2((स्थिर व्योम *) my_slot, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // All threads read variance, compute scale, apply to own chunk
        क्रम (पूर्णांक t = 0; t < workers; t++) {
            अस्थिर भग्न * slot = (अस्थिर भग्न *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            evict_to_l2((स्थिर व्योम *) slot, 1, 64);
        }
        WAIT_CACHEOPS;

        भग्न योग_var = 0.0f;
        क्रम (पूर्णांक t = 0; t < workers; t++) {
            अस्थिर भग्न * slot = (अस्थिर भग्न *) et_shire_l2scp_local((uint64_t) (group_base + t) * 64);
            योग_var += *slot;
        }

        स्थिर भग्न variance = et_fdiv(योग_var, (भग्न) (int32_t) ne0);
        स्थिर भग्न scale    = et_powf(variance + eps, -0.5f);

        अगर (!(scale > 0.0f)) {
            __asm__ अस्थिर("mova.m.x %0" ::"r"(saved_mask));
            वापस -1;
        }

        // Apply scale to centered values (already in dst from reduction 2)
        अगर (my_start < my_end) {
            uint32_t scale_bits;
            __asm__ अस्थिर("fmv.x.s %0, %1" : "=r"(scale_bits) : "f"(scale));
            __asm__ अस्थिर("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_bits) : "f13");

            क्रम (int32_t i0 = my_start; i0 < my_end; i0 += 8) {
                __asm__ अस्थिर(
                    "flw.ps f12, %[y_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fsw.ps f14, %[result]\n"
                    : [परिणाम] "=m"(*(भग्न (*)[8]) & dst_ptr[i0])
                    : [y_vec] "m"(*(स्थिर भग्न (*)[8]) & dst_ptr[i0])
                    : "f12", "f14");
            }
        }

        __asm__ अस्थिर("mova.m.x %0" ::"r"(saved_mask));
    }

    वापस 0;
}
