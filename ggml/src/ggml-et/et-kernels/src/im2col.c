//******************************************************************************
// IM2COL Kernel
// Baseline scalar implementation for:
//   src1: [N, IC, IH, IW] -> dst: [N, OH, OW, IC*KH*KW] (2D)
//   src1: [N, IC,    IW] -> dst: [N,  1, OW, IC*   KW] (1D)
//
// Work is distributed by row-groups so threads own cache-line-aligned chunks of
// destination rows even when ne0 is not cache aligned.
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ im2col_store_elem(ಶೂನ್ಯ * dst_ತಳ, ಎಣಿಕೆ ggml_type dst_type, int64_t idx, ದಶಮಾಂಶ ಮೌಲ್ಯ) {
    ಆದರೆ (dst_type == GGML_TYPE_F32) {
        ((ದಶಮಾಂಶ *) dst_ತಳ)[idx] = ಮೌಲ್ಯ;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ((uint16_t *) dst_ತಳ)[idx] = fp32_ಗಮ್ಯ_fp16(ಮೌಲ್ಯ);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ im2col_ತುಂಬು_src_elem(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src_ತಳ, ಎಣಿಕೆ ggml_type src_type, int64_t idx) {
    ಆದರೆ (src_type == GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src_ತಳ)[idx];
    }

    ಹಿಂತಿರುಗಿ fp16_ಗಮ್ಯ_fp32(((ಸ್ಥಿರಾಂಕ uint16_t *) src_ತಳ)[idx]);
}

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಆದರೆ (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಪೂರ್ಣಾಂಕ thread_id   = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಎಳೆಗಳು = get_ಅಂಕೆ_ಎಳೆಗಳು(kernel_env->shire_ಮುಖವಾಡ);

    ಆದರೆ (thread_id < 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ರಚನೆ ggml_tensor * src0 = &params->src0;
    ರಚನೆ ggml_tensor * src1 = &params->src1;
    ರಚನೆ ggml_tensor * dst  = &params->dst;

    ಆದರೆ (!src1->ದತ್ತಾಂಶ || !dst->ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಆದರೆ (!((dst->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) ||
          (dst->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32)))) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int32_t s0    = ((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[0];
    ಸ್ಥಿರಾಂಕ int32_t s1    = ((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[1];
    ಸ್ಥಿರಾಂಕ int32_t p0    = ((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[2];
    ಸ್ಥಿರಾಂಕ int32_t p1    = ((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[3];
    ಸ್ಥಿರಾಂಕ int32_t d0    = ((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[4];
    ಸ್ಥಿರಾಂಕ int32_t d1    = ((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[5];
    ಸ್ಥಿರಾಂಕ int32_t is_2d = ((ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params)[6];

    ಸ್ಥಿರಾಂಕ int64_t N  = is_2d ? src1->ne[3] : src1->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t IC = is_2d ? src1->ne[2] : src1->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t IH = is_2d ? src1->ne[1] : 1;
    ಸ್ಥಿರಾಂಕ int64_t IW = src1->ne[0];

    ಸ್ಥಿರಾಂಕ int64_t KH = is_2d ? src0->ne[1] : 1;
    ಸ್ಥಿರಾಂಕ int64_t KW = src0->ne[0];

    ಸ್ಥಿರಾಂಕ int64_t OH         = is_2d ? dst->ne[2] : 1;
    ಸ್ಥಿರಾಂಕ int64_t OW         = dst->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_elems  = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = OW * OH * N;

    ಸ್ಥಿರಾಂಕ size_t src_batch_stride   = is_2d ? src1->nb[3] : src1->nb[2];
    ಸ್ಥಿರಾಂಕ size_t src_channel_stride = is_2d ? src1->nb[2] : src1->nb[1];

    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_stride   = dst->nb[1];
    ಸ್ಥಿರಾಂಕ size_t dst_plane_stride = is_2d ? dst->nb[2] : 0;
    ಸ್ಥಿರಾಂಕ size_t dst_batch_stride = is_2d ? dst->nb[3] : dst->nb[2];

    ಸ್ಥಿರಾಂಕ int64_t dst_elem_ಗಾತ್ರ  = (dst->type == GGML_TYPE_F32) ? (int64_t) ಗಾತ್ರ(ದಶಮಾಂಶ) : (int64_t) ಗಾತ್ರ(uint16_t);
    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು = et_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_cacheline_ಗುಂಪು(ಅಡ್ಡಸಾಲು_elems, dst_elem_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_groups   = (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು - 1) / ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು;

    ಗಾಗಿ (int64_t grp = thread_id; grp < ಒಟ್ಟು_groups; grp += ಅಂಕೆ_ಎಳೆಗಳು) {
        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = grp * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು;
        int64_t       ಅಡ್ಡಸಾಲು_ಕೊನೆ   = ಅಡ್ಡಸಾಲು_ಆರಂಭ + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಗುಂಪು;
        ಆದರೆ (ಅಡ್ಡಸಾಲು_ಕೊನೆ > ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು) {
            ಅಡ್ಡಸಾಲು_ಕೊನೆ = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
        }

        ಗಾಗಿ (int64_t ಅಡ್ಡಸಾಲು = ಅಡ್ಡಸಾಲು_ಆರಂಭ; ಅಡ್ಡಸಾಲು < ಅಡ್ಡಸಾಲು_ಕೊನೆ; ++ಅಡ್ಡಸಾಲು) {
            ಸ್ಥಿರಾಂಕ int64_t ಒಳ  = ಅಡ್ಡಸಾಲು / (OH * OW);
            ಸ್ಥಿರಾಂಕ int64_t rem = ಅಡ್ಡಸಾಲು % (OH * OW);
            ಸ್ಥಿರಾಂಕ int64_t ioh = rem / OW;
            ಸ್ಥಿರಾಂಕ int64_t iow = rem % OW;

            ಶೂನ್ಯ * dst_ಅಡ್ಡಸಾಲು = (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ + ಒಳ * dst_batch_stride + ioh * dst_plane_stride + iow * dst_ಅಡ್ಡಸಾಲು_stride;

            ಗಾಗಿ (int64_t iic = 0; iic < IC; ++iic) {
                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src_channel = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src1->ದತ್ತಾಂಶ + ಒಳ * src_batch_stride + iic * src_channel_stride;

                ಗಾಗಿ (int64_t ikh = 0; ikh < KH; ++ikh) {
                    ಗಾಗಿ (int64_t ikw = 0; ikw < KW; ++ikw) {
                        ಸ್ಥಿರಾಂಕ int64_t iiw     = iow * s0 + ikw * d0 - p0;
                        ಸ್ಥಿರಾಂಕ int64_t iih     = ioh * s1 + ikh * d1 - p1;
                        ಸ್ಥಿರಾಂಕ int64_t dst_idx = iic * (KH * KW) + ikh * KW + ikw;

                        ಆದರೆ (iiw < 0 || iiw >= IW || iih < 0 || iih >= IH) {
                            im2col_store_elem(dst_ಅಡ್ಡಸಾಲು, dst->type, dst_idx, 0.0f);
                        } ಇಲ್ಲದಿದ್ದರೆ {
                            ಸ್ಥಿರಾಂಕ int64_t src_idx = iih * IW + iiw;
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   ಮೌಲ್ಯ   = im2col_ತುಂಬು_src_elem(src_channel, src1->type, src_idx);
                            im2col_store_elem(dst_ಅಡ್ಡಸಾಲು, dst->type, dst_idx, ಮೌಲ್ಯ);
                        }
                    }
                }
            }
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
