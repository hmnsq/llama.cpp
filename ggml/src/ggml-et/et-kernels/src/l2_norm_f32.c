//******************************************************************************
// L2 Norm F32 Kernel (L2 Normalization)
// y[i] = x[i] / max(sqrt(sum(x^2)), eps)
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <string.h>

// L2 Norm kernel parameters structure
ರಚನೆ ggml_et_l2_norm_params {
    ರಚನೆ ggml_tensor src0;  // F32 input tensor
    ರಚನೆ ggml_tensor dst;   // F32 output tensor
    ದಶಮಾಂಶ              eps;   // Epsilon parameter for numerical stability
};

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_l2_norm_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಆದರೆ (!kernel_env) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಪೂರ್ಣಾಂಕ thread_id   = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಎಳೆಗಳು = get_ಅಂಕೆ_ಎಳೆಗಳು(kernel_env->shire_ಮುಖವಾಡ);

    ಆದರೆ (thread_id < 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಆದರೆ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಹಿಂತಿರುಗಿ -1;  // Invalid pointer
    }

    ರಚನೆ ggml_tensor * src0 = &params->src0;
    ರಚನೆ ggml_tensor * dst  = &params->dst;
    ದಶಮಾಂಶ                eps  = params->eps;

    ಆದರೆ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;  // Unsupported type combination
    }

    ದಶಮಾಂಶ * src0_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಆದರೆ (!src0_ದತ್ತಾಂಶ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;  // Null data pointer
    }

    ಆದರೆ (eps < 0.0f) {
        ಹಿಂತಿರುಗಿ -1;  // Invalid epsilon
    }

    ಸ್ಥಿರಾಂಕ int64_t ne0 = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne1 = dst->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne2 = dst->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne3 = dst->ne[3];

    ಸ್ಥಿರಾಂಕ size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ಸ್ಥಿರಾಂಕ size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    (ಶೂನ್ಯ) nb0;
    (ಶೂನ್ಯ) nb00;

    ಆದರೆ (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        ಹಿಂತಿರುಗಿ -1;  // Shape mismatch
    }

    ಸ್ಥಿರಾಂಕ int32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು    = (int32_t) (ne1 * ne2 * ne3);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     shire_ಎಳೆಗಳು = SOC_MINIONS_ಪ್ರತಿ_SHIRE * ಅಂಕೆ_HARTS_ಪ್ರತಿ_MINION;

    ಆದರೆ (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು >= shire_ಎಳೆಗಳು) {
        // Row-parallel: each thread processes whole rows
        ಗಾಗಿ (int64_t i3 = 0; i3 < ne3; i3++) {
            ಗಾಗಿ (int64_t i2 = 0; i2 < ne2; i2++) {
                ಗಾಗಿ (int64_t i1 = thread_id; i1 < ne1; i1 += ಅಂಕೆ_ಎಳೆಗಳು) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ptr =
                        (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    ದಶಮಾಂಶ * dst_ptr = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + i3 * nb3 + i2 * nb2 + i1 * nb1);

                    ದಶಮಾಂಶ zero = 0.0f;
                    __asm__ ಅಸ್ಥಿರ("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                    ಗಾಗಿ (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ ಅಸ್ಥಿರ(
                            "flw.ps f11, %[x_vec]\n"
                            "fmadd.ps f10, f11, f11, f10\n"
                            :
                            : [x_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    ದಶಮಾಂಶ ಮೊತ್ತ_sq;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(ಮೊತ್ತ_sq)::"t0", "f1", "f2", "f3", "f4", "f5");

                    ದಶಮಾಂಶ l2_norm = et_powf(ಮೊತ್ತ_sq, 0.5f);
                    ಆದರೆ (l2_norm < eps) {
                        l2_norm = eps;
                    }
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = et_fdiv(1.0f, l2_norm);

                    ಗಾಗಿ (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ ಅಸ್ಥಿರ(
                            "flw.ps f11, %[x_vec]\n"
                            "fbc.ps f12, %[scale_ptr]\n"
                            "fmul.ps f13, f11, f12\n"
                            "fsw.ps f13, %[result]\n"
                            : [ಫಲಿತಾಂಶ] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src_ptr[i0]), [scale_ptr] "m"(scale)
                            : "f11", "f12", "f13");
                    }
                }
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Intra-row: threads within each shire cooperate via L2 SCP
        ಪೂರ್ಣಾಂಕ shire_tid       = thread_id % shire_ಎಳೆಗಳು;
        ಪೂರ್ಣಾಂಕ ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = shire_ಎಳೆಗಳು / ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
        ಪೂರ್ಣಾಂಕ my_ಅಡ್ಡಸಾಲು          = shire_tid / ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
        ಪೂರ್ಣಾಂಕ local_tid       = shire_tid % ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
        ಪೂರ್ಣಾಂಕ ಗುಂಪು_ತಳ      = my_ಅಡ್ಡಸಾಲು * ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

        ಆದರೆ (my_ಅಡ್ಡಸಾಲು >= ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            ಹಿಂತಿರುಗಿ 0;
        }

        int64_t i1 = my_ಅಡ್ಡಸಾಲು % ne1;
        int64_t i2 = (my_ಅಡ್ಡಸಾಲು / ne1) % ne2;
        int64_t i3 = my_ಅಡ್ಡಸಾಲು / (ne1 * ne2);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i3 * nb03 + i2 * nb02 + i1 * nb01);
        ದಶಮಾಂಶ *       dst_ptr = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + i3 * nb3 + i2 * nb2 + i1 * nb1);

        ಸ್ಥಿರಾಂಕ int32_t elems_ಪ್ರತಿ_cl   = 16;
        int32_t       ಒಟ್ಟು_cls      = ((int32_t) ne0 + elems_ಪ್ರತಿ_cl - 1) / elems_ಪ್ರತಿ_cl;
        int32_t       cls_ಪ್ರತಿ_thread = (ಒಟ್ಟು_cls + ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು - 1) / ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
        int32_t       my_ಆರಂಭ       = local_tid * cls_ಪ್ರತಿ_thread * elems_ಪ್ರತಿ_cl;
        int32_t       my_ಕೊನೆ         = my_ಆರಂಭ + cls_ಪ್ರತಿ_thread * elems_ಪ್ರತಿ_cl;
        ಆದರೆ (my_ಕೊನೆ > (int32_t) ne0) {
            my_ಕೊನೆ = (int32_t) ne0;
        }
        ಆದರೆ (my_ಆರಂಭ >= (int32_t) ne0) {
            my_ಆರಂಭ = 0;
            my_ಕೊನೆ   = 0;
        }

        ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ saved_ಮುಖವಾಡ;
        __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(saved_ಮುಖವಾಡ));
        __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");

        // Phase 1: partial sum of squares
        __asm__ ಅಸ್ಥಿರ("fbci.pi f10, 0" ::: "f10");
        ಗಾಗಿ (int32_t i0 = my_ಆರಂಭ; i0 < my_ಕೊನೆ; i0 += 8) {
            __asm__ ಅಸ್ಥಿರ(
                "flw.ps f11, %[x_vec]\n"
                "fmadd.ps f10, f11, f11, f10\n"
                :
                : [x_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src_ptr[i0])
                : "f10", "f11");
        }

        ದಶಮಾಂಶ partial_ಮೊತ್ತ;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(partial_ಮೊತ್ತ)::"t0", "f1", "f2", "f3", "f4", "f5");

        // Phase 2: L2SCP exchange
        ಅಸ್ಥಿರ ದಶಮಾಂಶ * my_ಸ್ಥಾನ = (ಅಸ್ಥಿರ ದಶಮಾಂಶ *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_ಸ್ಥಾನ                 = partial_ಮೊತ್ತ;
        FENCE;
        evict_ಗಮ್ಯ_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) my_ಸ್ಥಾನ, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // Phase 3: all threads reduce + apply scale to own chunk
        ಪೂರ್ಣಾಂಕ workers = ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು < ಒಟ್ಟು_cls ? ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು : ಒಟ್ಟು_cls;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < workers; t++) {
            ಅಸ್ಥಿರ ದಶಮಾಂಶ * ಸ್ಥಾನ = (ಅಸ್ಥಿರ ದಶಮಾಂಶ *) et_shire_l2scp_local((uint64_t) (ಗುಂಪು_ತಳ + t) * 64);
            evict_ಗಮ್ಯ_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) ಸ್ಥಾನ, 1, 64);
        }
        WAIT_CACHEOPS;

        ದಶಮಾಂಶ ಒಟ್ಟು_ಮೊತ್ತ_sq = 0.0f;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < workers; t++) {
            ಅಸ್ಥಿರ ದಶಮಾಂಶ * ಸ್ಥಾನ = (ಅಸ್ಥಿರ ದಶಮಾಂಶ *) et_shire_l2scp_local((uint64_t) (ಗುಂಪು_ತಳ + t) * 64);
            ಒಟ್ಟು_ಮೊತ್ತ_sq += *ಸ್ಥಾನ;
        }

        ದಶಮಾಂಶ l2_norm = et_powf(ಒಟ್ಟು_ಮೊತ್ತ_sq, 0.5f);
        ಆದರೆ (l2_norm < eps) {
            l2_norm = eps;
        }
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = et_fdiv(1.0f, l2_norm);

        ಆದರೆ (my_ಆರಂಭ < my_ಕೊನೆ) {
            uint32_t scale_ಬಿಟ್ಟುಗಳು;
            __asm__ ಅಸ್ಥಿರ("fmv.x.s %0, %1" : "=r"(scale_ಬಿಟ್ಟುಗಳು) : "f"(scale));
            __asm__ ಅಸ್ಥಿರ("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_ಬಿಟ್ಟುಗಳು) : "f13");

            ಗಾಗಿ (int32_t i0 = my_ಆರಂಭ; i0 < my_ಕೊನೆ; i0 += 8) {
                __asm__ ಅಸ್ಥಿರ(
                    "flw.ps f12, %[x_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fsw.ps f14, %[result]\n"
                    : [ಫಲಿತಾಂಶ] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src_ptr[i0])
                    : "f12", "f14");
            }
        }

        __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(saved_ಮುಖವಾಡ));
    }

    ಹಿಂತಿರುಗಿ 0;
}
