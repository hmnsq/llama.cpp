//******************************************************************************
// Concat F32 Kernel
// Concatenates two F32 tensors along a specified dimension.
// All copies are aligned to cacheline boundaries (64 bytes = 16 floats).
//
// For dim >= 1, entire rows are copied from src0 or src1 into dst.
// For dim == 0, use:
// - a fast vector path when both source row segments are cacheline-aligned
// - a scalar stride-aware path otherwise
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <string.h>

ರಚನೆ ggml_et_concat_params {
    ರಚನೆ ggml_tensor src0;  // F32 input tensor 0
    ರಚನೆ ggml_tensor src1;  // F32 input tensor 1
    ರಚನೆ ggml_tensor dst;   // F32 output tensor
    int32_t            dim;   // Concatenation dimension
};

// Copy n floats from src to dst using 8-wide vector loads/stores.
// n must be a multiple of 16 (cacheline-aligned).
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ copy_ಅಡ್ಡಸಾಲು_aligned(ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, int32_t n) {
    ಗಾಗಿ (int32_t i = 0; i < n; i += 8) {
        __asm__ ಅಸ್ಥಿರ(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst[i])
            : [src_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & src[i])
            : "f11");
    }
}

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_concat_params * params, ಶೂನ್ಯ * env) {
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

    ರಚನೆ ggml_tensor * src0 = &params->src0;
    ರಚನೆ ggml_tensor * src1 = &params->src1;
    ರಚನೆ ggml_tensor * dst  = &params->dst;
    int32_t              dim  = params->dim;

    ಆದರೆ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ದಶಮಾಂಶ * src0_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * src1_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಆದರೆ (!src0_ದತ್ತಾಂಶ || !src1_ದತ್ತಾಂಶ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    ಸ್ಥಿರಾಂಕ int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    ಸ್ಥಿರಾಂಕ int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];

    // src strides in bytes
    ಸ್ಥಿರಾಂಕ size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    ಸ್ಥಿರಾಂಕ size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    // dst strides in bytes
    ಸ್ಥಿರಾಂಕ size_t dnb1 = dst->nb[1], dnb2 = dst->nb[2], dnb3 = dst->nb[3];

    // Total rows across all higher dimensions
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = ne1 * ne2 * ne3;

    // Generic slow path for dim==0 when either source segment is not suitable for
    // aligned vector copies. Threading is done by cacheline-aligned row groups,
    // so writers do not share destination cache lines.
    ಆದರೆ (dim == 0 && (ne00 % 16 != 0 || ne10 % 16 != 0 || nb00 != ಗಾತ್ರ(ದಶಮಾಂಶ) || nb10 != ಗಾತ್ರ(ದಶಮಾಂಶ))) {
        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು = et_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_cacheline_ಗುಂಪು(ne0, ಗಾತ್ರ(ದಶಮಾಂಶ));
        ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_groups   = (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು - 1) / ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು;

        ಗಾಗಿ (int64_t grp = thread_id; grp < ಒಟ್ಟು_groups; grp += ಅಂಕೆ_ಎಳೆಗಳು) {
            ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = grp * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು;
            int64_t       ಅಡ್ಡಸಾಲು_ಕೊನೆ   = ಅಡ್ಡಸಾಲು_ಆರಂಭ + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು;
            ಆದರೆ (ಅಡ್ಡಸಾಲು_ಕೊನೆ > ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು) {
                ಅಡ್ಡಸಾಲು_ಕೊನೆ = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
            }

            ಗಾಗಿ (int64_t ಅಡ್ಡಸಾಲು = ಅಡ್ಡಸಾಲು_ಆರಂಭ; ಅಡ್ಡಸಾಲು < ಅಡ್ಡಸಾಲು_ಕೊನೆ; ಅಡ್ಡಸಾಲು++) {
                int64_t i1 = ಅಡ್ಡಸಾಲು % ne1;
                int64_t i2 = (ಅಡ್ಡಸಾಲು / ne1) % ne2;
                int64_t i3 = ಅಡ್ಡಸಾಲು / (ne1 * ne2);

                ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * s0_ತಳ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i1 * nb01 + i2 * nb02 + i3 * nb03;
                ಗಾಗಿ (int64_t i0 = 0; i0 < ne00; i0++) {
                    dst_ಅಡ್ಡಸಾಲು[i0] = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (s0_ತಳ + i0 * nb00);
                }

                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * s1_ತಳ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + i1 * nb11 + i2 * nb12 + i3 * nb13;
                ಗಾಗಿ (int64_t i0 = 0; i0 < ne10; i0++) {
                    dst_ಅಡ್ಡಸಾಲು[ne00 + i0] = *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (s1_ತಳ + i0 * nb10);
                }
            }
        }
        ಹಿಂತಿರುಗಿ 0;
    }

    // Standard path: ne0 % 16 == 0, aligned rows
    ಗಾಗಿ (int64_t ಅಡ್ಡಸಾಲು = thread_id; ಅಡ್ಡಸಾಲು < ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು; ಅಡ್ಡಸಾಲು += ಅಂಕೆ_ಎಳೆಗಳು) {
        // Decompose linear row index into (i1, i2, i3)
        int64_t i1 = ಅಡ್ಡಸಾಲು % ne1;
        int64_t i2 = (ಅಡ್ಡಸಾಲು / ne1) % ne2;
        int64_t i3 = ಅಡ್ಡಸಾಲು / (ne1 * ne2);

        ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        ಆದರೆ (dim == 0) {
            // Concat along innermost dimension: [src0_row | src1_row]
            // Both ne00 and ne10 are multiples of 16 (cacheline-aligned)
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s0_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i1 * nb01 + i2 * nb02 + i3 * nb03);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s1_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + i1 * nb11 + i2 * nb12 + i3 * nb13);

            copy_ಅಡ್ಡಸಾಲು_aligned(dst_ಅಡ್ಡಸಾಲು, s0_ಅಡ್ಡಸಾಲು, (int32_t) ne00);
            copy_ಅಡ್ಡಸಾಲು_aligned(dst_ಅಡ್ಡಸಾಲು + ne00, s1_ಅಡ್ಡಸಾಲು, (int32_t) ne10);

        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dim == 1) {
            // Concat along dim 1: first ne01 rows from src0, rest from src1
            ಆದರೆ (i1 < ne01) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s0_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_ಅಡ್ಡಸಾಲು_aligned(dst_ಅಡ್ಡಸಾಲು, s0_ಅಡ್ಡಸಾಲು, (int32_t) ne0);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s1_ಅಡ್ಡಸಾಲು =
                    (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + (i1 - ne01) * nb11 + i2 * nb12 + i3 * nb13);
                copy_ಅಡ್ಡಸಾಲು_aligned(dst_ಅಡ್ಡಸಾಲು, s1_ಅಡ್ಡಸಾಲು, (int32_t) ne0);
            }

        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dim == 2) {
            // Concat along dim 2: first ne02 slices from src0, rest from src1
            ಆದರೆ (i2 < ne02) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s0_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_ಅಡ್ಡಸಾಲು_aligned(dst_ಅಡ್ಡಸಾಲು, s0_ಅಡ್ಡಸಾಲು, (int32_t) ne0);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s1_ಅಡ್ಡಸಾಲು =
                    (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + i1 * nb11 + (i2 - ne02) * nb12 + i3 * nb13);
                copy_ಅಡ್ಡಸಾಲು_aligned(dst_ಅಡ್ಡಸಾಲು, s1_ಅಡ್ಡಸಾಲು, (int32_t) ne0);
            }

        } ಇಲ್ಲದಿದ್ದರೆ {
            // dim == 3: first ne03 batches from src0, rest from src1
            ಆದರೆ (i3 < ne03) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s0_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_ಅಡ್ಡಸಾಲು_aligned(dst_ಅಡ್ಡಸಾಲು, s0_ಅಡ್ಡಸಾಲು, (int32_t) ne0);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s1_ಅಡ್ಡಸಾಲು =
                    (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + i1 * nb11 + i2 * nb12 + (i3 - ne03) * nb13);
                copy_ಅಡ್ಡಸಾಲು_aligned(dst_ಅಡ್ಡಸಾಲು, s1_ಅಡ್ಡಸಾಲು, (int32_t) ne0);
            }
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
