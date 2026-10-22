//******************************************************************************
// Tri F32 Kernel
// Triangular masking: zero out elements outside the triangular region.
//
// tri_type (matches ggml_tri_type enum):
//   0 = UPPER_DIAG:  keep where i0 >= i1
//   1 = UPPER:       keep where i0 >  i1
//   2 = LOWER_DIAG:  keep where i0 <= i1
//   3 = LOWER:       keep where i0 <  i1
//
// Distribution: cache-line aligned chunks of the flat contiguous dst.
// Each element is individually classified as keep or zero based on its
// (i0, i1) coordinates. This avoids cache-line sharing between threads
// when ne0 is not a multiple of 16.
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

#ವ್ಯಾಖ್ಯಾನಿಸು TRI_TYPE_ಊರ್ಧ್ವ_DIAG 0
#ವ್ಯಾಖ್ಯಾನಿಸು TRI_TYPE_ಊರ್ಧ್ವ      1
#ವ್ಯಾಖ್ಯಾನಿಸು TRI_TYPE_ಅಧೋ_DIAG 2
#ವ್ಯಾಖ್ಯಾನಿಸು TRI_TYPE_ಅಧೋ      3

ರಚನೆ ggml_et_tri_params {
    ರಚನೆ ggml_tensor src0;
    ರಚನೆ ggml_tensor dst;
    int32_t            tri_type;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ keep_element(int32_t tri_type, int64_t i0, int64_t i1) {
    ಆಯ್ಕೆ (tri_type) {
        ಸ್ಥಿತಿ TRI_TYPE_ಅಧೋ:
            ಹಿಂತಿರುಗಿ i0 < i1;
        ಸ್ಥಿತಿ TRI_TYPE_ಅಧೋ_DIAG:
            ಹಿಂತಿರುಗಿ i0 <= i1;
        ಸ್ಥಿತಿ TRI_TYPE_ಊರ್ಧ್ವ:
            ಹಿಂತಿರುಗಿ i0 > i1;
        ಸ್ಥಿತಿ TRI_TYPE_ಊರ್ಧ್ವ_DIAG:
            ಹಿಂತಿರುಗಿ i0 >= i1;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ 0;
    }
}

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_tri_params * params, ಶೂನ್ಯ * env) {
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
        ಹಿಂತಿರುಗಿ -1;
    }

    ರಚನೆ ggml_tensor * src0     = &params->src0;
    ರಚನೆ ggml_tensor * dst      = &params->dst;
    int32_t              tri_type = params->tri_type;

    ಆದರೆ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ದಶಮಾಂಶ * src0_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಆದರೆ (!src0_ದತ್ತಾಂಶ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t ne0 = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne1 = dst->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne2 = dst->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne3 = dst->ne[3];

    ಸ್ಥಿರಾಂಕ size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ಸ್ಥಿರಾಂಕ size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = ne1 * ne2 * ne3;

    //==========================================================================
    // Fast path: ne0 % 16 == 0 — rows are cache-line aligned, distribute rows
    //==========================================================================
    ಆದರೆ (ne0 % 16 == 0) {
        ದಶಮಾಂಶ zero = 0.0f;
        __asm__ ಅಸ್ಥಿರ("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        ಗಾಗಿ (int64_t ಅಡ್ಡಸಾಲು = thread_id; ಅಡ್ಡಸಾಲು < ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು; ಅಡ್ಡಸಾಲು += ಅಂಕೆ_ಎಳೆಗಳು) {
            ಸ್ಥಿರಾಂಕ int64_t i1 = ಅಡ್ಡಸಾಲು % ne1;
            ಸ್ಥಿರಾಂಕ int64_t i2 = (ಅಡ್ಡಸಾಲು / ne1) % ne2;
            ಸ್ಥಿರಾಂಕ int64_t i3 = ಅಡ್ಡಸಾಲು / (ne1 * ne2);

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i1 * nb01 + i2 * nb02 + i3 * nb03);
            ದಶಮಾಂಶ *       dst_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + i1 * nb1 + i2 * nb2 + i3 * nb3);

            int64_t keep_ಆರಂಭ, keep_ಕೊನೆ;
            ಆಯ್ಕೆ (tri_type) {
                ಸ್ಥಿತಿ TRI_TYPE_ಅಧೋ:
                    keep_ಆರಂಭ = 0;
                    keep_ಕೊನೆ   = i1;
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ TRI_TYPE_ಅಧೋ_DIAG:
                    keep_ಆರಂಭ = 0;
                    keep_ಕೊನೆ   = i1 + 1;
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ TRI_TYPE_ಊರ್ಧ್ವ:
                    keep_ಆರಂಭ = i1 + 1;
                    keep_ಕೊನೆ   = ne0;
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ TRI_TYPE_ಊರ್ಧ್ವ_DIAG:
                    keep_ಆರಂಭ = i1;
                    keep_ಕೊನೆ   = ne0;
                    ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ -1;
            }
            ಆದರೆ (keep_ಕೊನೆ > ne0) {
                keep_ಕೊನೆ = ne0;
            }

            // Zero prefix [0, keep_start) — SIMD for aligned blocks, scalar tail
            int64_t i0 = 0;
            ಗಾಗಿ (; i0 + 8 <= keep_ಆರಂಭ; i0 += 8) {
                __asm__ ಅಸ್ಥಿರ("fsw.ps f10, %[d]\n" : [d] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ಅಡ್ಡಸಾಲು[i0])::"f10");
            }
            ಗಾಗಿ (; i0 < keep_ಆರಂಭ; i0++) {
                dst_ಅಡ್ಡಸಾಲು[i0] = 0.0f;
            }

            // Copy kept region [keep_start, keep_end) — SIMD + scalar tail
            ಗಾಗಿ (; i0 + 8 <= keep_ಕೊನೆ; i0 += 8) {
                __asm__ ಅಸ್ಥಿರ(
                    "flw.ps f11, %[s]\n"
                    "fsw.ps f11, %[d]\n"
                    : [d] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ಅಡ್ಡಸಾಲು[i0])
                    : [s] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src_ಅಡ್ಡಸಾಲು[i0])
                    : "f11");
            }
            ಗಾಗಿ (; i0 < keep_ಕೊನೆ; i0++) {
                dst_ಅಡ್ಡಸಾಲು[i0] = src_ಅಡ್ಡಸಾಲು[i0];
            }

            // Zero suffix [keep_end, ne0) — SIMD + scalar tail
            ಗಾಗಿ (; i0 + 8 <= ne0; i0 += 8) {
                __asm__ ಅಸ್ಥಿರ("fsw.ps f10, %[d]\n" : [d] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ಅಡ್ಡಸಾಲು[i0])::"f10");
            }
            ಗಾಗಿ (; i0 < ne0; i0++) {
                dst_ಅಡ್ಡಸಾಲು[i0] = 0.0f;
            }
        }
        ಹಿಂತಿರುಗಿ 0;
    }

    //==========================================================================
    // Unaligned fallback: distribute by cache lines, scalar per element
    //==========================================================================
    {
        ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_elements = ne0 * ne1 * ne2 * ne3;
        ಸ್ಥಿರಾಂಕ int64_t elems_ಪ್ರತಿ_cl   = 16;
        ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_cl       = (ಒಟ್ಟು_elements + elems_ಪ್ರತಿ_cl - 1) / elems_ಪ್ರತಿ_cl;

        ಸ್ಥಿರಾಂಕ int64_t cl_ಪ್ರತಿ_thread = (ಒಟ್ಟು_cl + ಅಂಕೆ_ಎಳೆಗಳು - 1) / ಅಂಕೆ_ಎಳೆಗಳು;
        ಸ್ಥಿರಾಂಕ int64_t cl_ಆರಂಭ      = thread_id * cl_ಪ್ರತಿ_thread;
        int64_t       cl_ಕೊನೆ        = cl_ಆರಂಭ + cl_ಪ್ರತಿ_thread;
        ಆದರೆ (cl_ಕೊನೆ > ಒಟ್ಟು_cl) {
            cl_ಕೊನೆ = ಒಟ್ಟು_cl;
        }
        ಆದರೆ (cl_ಆರಂಭ >= ಒಟ್ಟು_cl) {
            ಹಿಂತಿರುಗಿ 0;
        }

        ಸ್ಥಿರಾಂಕ int64_t es = cl_ಆರಂಭ * elems_ಪ್ರತಿ_cl;
        int64_t       ee = cl_ಕೊನೆ * elems_ಪ್ರತಿ_cl;
        ಆದರೆ (ee > ಒಟ್ಟು_elements) {
            ee = ಒಟ್ಟು_elements;
        }

        int64_t ಅಡ್ಡಸಾಲು_idx = es / ne0;
        int64_t ಲಂಬಸಾಲು     = es % ne0;

        int64_t pos = es;
        ಆಗ (pos < ee) {
            ಸ್ಥಿರಾಂಕ int64_t i1 = ಅಡ್ಡಸಾಲು_idx % ne1;
            ಸ್ಥಿರಾಂಕ int64_t i2 = (ಅಡ್ಡಸಾಲು_idx / ne1) % ne2;
            ಸ್ಥಿರಾಂಕ int64_t i3 = ಅಡ್ಡಸಾಲು_idx / (ne1 * ne2);

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i1 * nb01 + i2 * nb02 + i3 * nb03);

            int64_t ಅಡ್ಡಸಾಲು_ಉಳಿದ   = ne0 - ಲಂಬಸಾಲು;
            int64_t chunk_ಉಳಿದ = ee - pos;
            int64_t n               = ಅಡ್ಡಸಾಲು_ಉಳಿದ < chunk_ಉಳಿದ ? ಅಡ್ಡಸಾಲು_ಉಳಿದ : chunk_ಉಳಿದ;

            int64_t keep_ಆರಂಭ, keep_ಕೊನೆ;
            ಆಯ್ಕೆ (tri_type) {
                ಸ್ಥಿತಿ TRI_TYPE_ಅಧೋ:
                    keep_ಆರಂಭ = 0;
                    keep_ಕೊನೆ   = i1;
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ TRI_TYPE_ಅಧೋ_DIAG:
                    keep_ಆರಂಭ = 0;
                    keep_ಕೊನೆ   = i1 + 1;
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ TRI_TYPE_ಊರ್ಧ್ವ:
                    keep_ಆರಂಭ = i1 + 1;
                    keep_ಕೊನೆ   = ne0;
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ TRI_TYPE_ಊರ್ಧ್ವ_DIAG:
                    keep_ಆರಂಭ = i1;
                    keep_ಕೊನೆ   = ne0;
                    ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ -1;
            }
            ಆದರೆ (keep_ಕೊನೆ > ne0) {
                keep_ಕೊನೆ = ne0;
            }

            int64_t ಕೊನೆ_ಲಂಬಸಾಲು = ಲಂಬಸಾಲು + n;
            ಗಾಗಿ (int64_t i0 = ಲಂಬಸಾಲು; i0 < ಕೊನೆ_ಲಂಬಸಾಲು; i0++) {
                ಆದರೆ (i0 >= keep_ಆರಂಭ && i0 < keep_ಕೊನೆ) {
                    dst_ದತ್ತಾಂಶ[pos + (i0 - ಲಂಬಸಾಲು)] = src_ಅಡ್ಡಸಾಲು[i0];
                } ಇಲ್ಲದಿದ್ದರೆ {
                    dst_ದತ್ತಾಂಶ[pos + (i0 - ಲಂಬಸಾಲು)] = 0.0f;
                }
            }

            pos += n;
            ಲಂಬಸಾಲು = 0;
            ಅಡ್ಡಸಾಲು_idx++;
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
