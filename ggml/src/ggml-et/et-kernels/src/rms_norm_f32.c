//******************************************************************************
// RMS Norm F32 Kernel
// Root Mean Square normalization: y[i] = x[i] / sqrt(mean(x^2) + eps)
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <string.h>

// RMS norm kernel parameters structure
ರಚನೆ ggml_et_rms_norm_params {
    ರಚನೆ ggml_tensor src0;  // F32 input tensor
    ರಚನೆ ggml_tensor dst;   // F32 output tensor
    ದಶಮಾಂಶ              eps;   // Epsilon parameter for numerical stability
};

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_rms_norm_params * params, ಶೂನ್ಯ * env) {
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

    ಸ್ಥಿರಾಂಕ int64_t ne0 = dst->ne[0];  // Inner dimension (row size)
    ಸ್ಥಿರಾಂಕ int64_t ne1 = dst->ne[1];  // Dimension 1
    ಸ್ಥಿರಾಂಕ int64_t ne2 = dst->ne[2];  // Dimension 2
    ಸ್ಥಿರಾಂಕ int64_t ne3 = dst->ne[3];  // Dimension 3

    // Get dst strides (in bytes)
    ಸ್ಥಿರಾಂಕ size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    // Get src0 strides (in bytes)
    ಸ್ಥಿರಾಂಕ size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Verify that src0 and dst have same shape (required for RMS norm)
    ಆದರೆ (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        ಹಿಂತಿರುಗಿ -1;  // Shape mismatch
    }

    // RMS norm processes rows independently
    // Parallelize across rows using simple striding
    // TODO: ensure lines don't cross cache lines
    // Precompute reciprocal of row length (constant across all rows)
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   inv_ne0    = et_fdiv(1.0f, (ದಶಮಾಂಶ) (int32_t) ne0);
    ಸ್ಥಿರಾಂಕ int32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = (int32_t) (ne1 * ne2 * ne3);

    // Intra-row cooperation only works within a single shire (barrier + L2SCP
    // are shire-local). Use per-shire thread count for the threshold.
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ shire_ಎಳೆಗಳು = SOC_MINIONS_ಪ್ರತಿ_SHIRE * ಅಂಕೆ_HARTS_ಪ್ರತಿ_MINION;  // 64

    ಆದರೆ (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು >= shire_ಎಳೆಗಳು) {
        // Row-parallel: each thread processes whole rows
        ಗಾಗಿ (int64_t i3 = 0; i3 < ne3; i3++) {
            ಗಾಗಿ (int64_t i2 = 0; i2 < ne2; i2++) {
                ಗಾಗಿ (int64_t i1 = thread_id; i1 < ne1; i1 += ಅಂಕೆ_ಎಳೆಗಳು) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ptr =
                        (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    ದಶಮಾಂಶ * dst_ptr = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + i3 * nb3 + i2 * nb2 + i1 * nb1);

                    // Set mask to enable all 8 vector lanes
                    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ saved_ಮುಖವಾಡ;
                    __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(saved_ಮುಖವಾಡ));
                    __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");

                    // Step 1: Compute sum of squares using 8-wide vectors
                    __asm__ ಅಸ್ಥಿರ("fbci.pi f10, 0" ::: "f10");

                    ಗಾಗಿ (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ ಅಸ್ಥಿರ(
                            "flw.ps f11, %[x_vec]\n"
                            "fmadd.ps f10, f11, f11, f10\n"
                            :
                            : [x_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    // Horizontal reduce
                    ದಶಮಾಂಶ ಮೊತ್ತ;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(ಮೊತ್ತ)::"t0", "f1", "f2", "f3", "f4", "f5");

                    // Step 2: scale = rsqrt(mean + eps)
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = et_powf(ಮೊತ್ತ * inv_ne0 + eps, -0.5f);

                    ಆದರೆ (!(scale > 0.0f)) {
                        __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(saved_ಮುಖವಾಡ));
                        ಹಿಂತಿರುಗಿ -1;
                    }

                    // Step 3: Apply scaling: broadcast scale once, reuse across loop
                    uint32_t scale_ಬಿಟ್ಟುಗಳು;
                    __asm__ ಅಸ್ಥಿರ("fmv.x.s %0, %1" : "=r"(scale_ಬಿಟ್ಟುಗಳು) : "f"(scale));
                    __asm__ ಅಸ್ಥಿರ("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_ಬಿಟ್ಟುಗಳು) : "f13");

                    ಗಾಗಿ (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ ಅಸ್ಥಿರ(
                            "flw.ps f12, %[x_vec]\n"
                            "fmul.ps f14, f12, f13\n"
                            "fsw.ps f14, %[result]\n"
                            : [ಫಲಿತಾಂಶ] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src_ptr[i0])
                            : "f12", "f14");
                    }

                    __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(saved_ಮುಖವಾಡ));
                }
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Intra-row: threads within each shire cooperate on rows via L2 SCP.
        // L2 SCP + barrier are shire-local, so use shire-local thread index.
        ಪೂರ್ಣಾಂಕ shire_tid       = thread_id % shire_ಎಳೆಗಳು;  // 0..63 within this shire
        ಪೂರ್ಣಾಂಕ ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = shire_ಎಳೆಗಳು / ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
        ಪೂರ್ಣಾಂಕ my_ಅಡ್ಡಸಾಲು          = shire_tid / ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
        ಪೂರ್ಣಾಂಕ local_tid       = shire_tid % ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
        ಪೂರ್ಣಾಂಕ ಗುಂಪು_ತಳ      = my_ಅಡ್ಡಸಾಲು * ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;  // shire-local group base

        // Excess threads within this shire, barrier and leave
        ಆದರೆ (my_ಅಡ್ಡಸಾಲು >= ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            ಹಿಂತಿರುಗಿ 0;
        }

        // Unflatten row index
        int64_t i1 = my_ಅಡ್ಡಸಾಲು % ne1;
        int64_t i2 = (my_ಅಡ್ಡಸಾಲು / ne1) % ne2;
        int64_t i3 = my_ಅಡ್ಡಸಾಲು / (ne1 * ne2);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ptr = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i3 * nb03 + i2 * nb02 + i1 * nb01);
        ದಶಮಾಂಶ *       dst_ptr = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + i3 * nb3 + i2 * nb2 + i1 * nb1);

        // Chunk boundaries aligned to 16 floats (64-byte cache line)
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

        // Phase 1: each thread computes partial sum of squares on its chunk
        __asm__ ಅಸ್ಥಿರ("fbci.pi f10, 0" ::: "f10");
        ಗಾಗಿ (int32_t i0 = my_ಆರಂಭ; i0 < my_ಕೊನೆ; i0 += 8) {
            __asm__ ಅಸ್ಥಿರ(
                "flw.ps f11, %[x_vec]\n"
                "fmadd.ps f10, f11, f11, f10\n"
                :
                : [x_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src_ptr[i0])
                : "f10", "f11");
        }

        // Horizontal reduce to scalar
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

        // Phase 2: write partial sum to L2 SCP, evict from L1D
        ಅಸ್ಥಿರ ದಶಮಾಂಶ * my_ಸ್ಥಾನ = (ಅಸ್ಥಿರ ದಶಮಾಂಶ *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_ಸ್ಥಾನ                 = partial_ಮೊತ್ತ;
        FENCE;
        evict_ಗಮ್ಯ_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) my_ಸ್ಥಾನ, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // Phase 3: ALL threads read partial sums, compute scale, apply to own chunk.
        // Each thread independently reduces to avoid a second barrier.
        ಪೂರ್ಣಾಂಕ workers = ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು < ಒಟ್ಟು_cls ? ಎಳೆಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು : ಒಟ್ಟು_cls;

        // Evict stale L1D entries for worker slots
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < workers; t++) {
            ಅಸ್ಥಿರ ದಶಮಾಂಶ * ಸ್ಥಾನ = (ಅಸ್ಥಿರ ದಶಮಾಂಶ *) et_shire_l2scp_local((uint64_t) (ಗುಂಪು_ತಳ + t) * 64);
            evict_ಗಮ್ಯ_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) ಸ್ಥಾನ, 1, 64);
        }
        WAIT_CACHEOPS;

        // Every thread reduces the same partial sums -> same scale
        ದಶಮಾಂಶ ಒಟ್ಟು_ಮೊತ್ತ = 0.0f;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < workers; t++) {
            ಅಸ್ಥಿರ ದಶಮಾಂಶ * ಸ್ಥಾನ = (ಅಸ್ಥಿರ ದಶಮಾಂಶ *) et_shire_l2scp_local((uint64_t) (ಗುಂಪು_ತಳ + t) * 64);
            ಒಟ್ಟು_ಮೊತ್ತ += *ಸ್ಥಾನ;
        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = et_powf(ಒಟ್ಟು_ಮೊತ್ತ * inv_ne0 + eps, -0.5f);
        ಆದರೆ (!(scale > 0.0f)) {
            __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(saved_ಮುಖವಾಡ));
            ಹಿಂತಿರುಗಿ -1;
        }

        // Each thread applies scale to its own chunk only
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
