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

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ im2col_store_elem(ಶೂನ್ಯ * dst_base, ಗಣನಾ ggml_type dst_type, int64_t idx, ಭಿನ್ನ ಮೌಲ್ಯ) {
    ಯದಿ (dst_type == GGML_TYPE_F32) {
        ((ಭಿನ್ನ *) dst_base)[idx] = ಮೌಲ್ಯ;
    } ಅನ್ಯಥಾ {
        ((uint16_t *) dst_base)[idx] = fp32_to_fp16(ಮೌಲ್ಯ);
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ im2col_load_src_elem(ಧ್ರುವ ಶೂನ್ಯ * src_base, ಗಣನಾ ggml_type src_type, int64_t idx) {
    ಯದಿ (src_type == GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ((ಧ್ರುವ ಭಿನ್ನ *) src_base)[idx];
    }

    ಪ್ರತ್ಯಾಗಮ fp16_to_fp32(((ಧ್ರುವ uint16_t *) src_base)[idx]);
}

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_ದ್ವಿಮಾನ_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಯದಿ (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪೂರ್ಣಾಙ್ಕ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_threads = get_ಅಂಕೆ_threads(kernel_env->shire_mask);

    ಯದಿ (thread_id < 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಸಂರಚನಾ ggml_tensor * src0 = &params->src0;
    ಸಂರಚನಾ ggml_tensor * src1 = &params->src1;
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;

    ಯದಿ (!src1->data || !dst->data) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಯದಿ (!((dst->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) ||
          (dst->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32)))) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ int32_t s0    = ((ಧ್ರುವ int32_t *) dst->op_params)[0];
    ಧ್ರುವ int32_t s1    = ((ಧ್ರುವ int32_t *) dst->op_params)[1];
    ಧ್ರುವ int32_t p0    = ((ಧ್ರುವ int32_t *) dst->op_params)[2];
    ಧ್ರುವ int32_t p1    = ((ಧ್ರುವ int32_t *) dst->op_params)[3];
    ಧ್ರುವ int32_t d0    = ((ಧ್ರುವ int32_t *) dst->op_params)[4];
    ಧ್ರುವ int32_t d1    = ((ಧ್ರುವ int32_t *) dst->op_params)[5];
    ಧ್ರುವ int32_t is_2d = ((ಧ್ರುವ int32_t *) dst->op_params)[6];

    ಧ್ರುವ int64_t N  = is_2d ? src1->ne[3] : src1->ne[2];
    ಧ್ರುವ int64_t IC = is_2d ? src1->ne[2] : src1->ne[1];
    ಧ್ರುವ int64_t IH = is_2d ? src1->ne[1] : 1;
    ಧ್ರುವ int64_t IW = src1->ne[0];

    ಧ್ರುವ int64_t KH = is_2d ? src0->ne[1] : 1;
    ಧ್ರುವ int64_t KW = src0->ne[0];

    ಧ್ರುವ int64_t OH         = is_2d ? dst->ne[2] : 1;
    ಧ್ರುವ int64_t OW         = dst->ne[1];
    ಧ್ರುವ int64_t row_elems  = dst->ne[0];
    ಧ್ರುವ int64_t total_rows = OW * OH * N;

    ಧ್ರುವ size_t src_batch_stride   = is_2d ? src1->nb[3] : src1->nb[2];
    ಧ್ರುವ size_t src_channel_stride = is_2d ? src1->nb[2] : src1->nb[1];

    ಧ್ರುವ size_t dst_row_stride   = dst->nb[1];
    ಧ್ರುವ size_t dst_plane_stride = is_2d ? dst->nb[2] : 0;
    ಧ್ರುವ size_t dst_batch_stride = is_2d ? dst->nb[3] : dst->nb[2];

    ಧ್ರುವ int64_t dst_elem_size  = (dst->type == GGML_TYPE_F32) ? (int64_t) ಪರಿಮಾಣ(ಭಿನ್ನ) : (int64_t) ಪರಿಮಾಣ(uint16_t);
    ಧ್ರುವ int64_t rows_per_group = et_rows_per_cacheline_group(row_elems, dst_elem_size);
    ಧ್ರುವ int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

    ಪ್ರತಿ (int64_t grp = thread_id; grp < total_groups; grp += ಅಂಕೆ_threads) {
        ಧ್ರುವ int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        ಯದಿ (row_end > total_rows) {
            row_end = total_rows;
        }

        ಪ್ರತಿ (int64_t row = row_start; row < row_end; ++row) {
            ಧ್ರುವ int64_t in  = row / (OH * OW);
            ಧ್ರುವ int64_t rem = row % (OH * OW);
            ಧ್ರುವ int64_t ioh = rem / OW;
            ಧ್ರುವ int64_t iow = rem % OW;

            ಶೂನ್ಯ * dst_row = (ವರ್ಣ *) dst->data + in * dst_batch_stride + ioh * dst_plane_stride + iow * dst_row_stride;

            ಪ್ರತಿ (int64_t iic = 0; iic < IC; ++iic) {
                ಧ್ರುವ ಶೂನ್ಯ * src_channel = (ಧ್ರುವ ವರ್ಣ *) src1->data + in * src_batch_stride + iic * src_channel_stride;

                ಪ್ರತಿ (int64_t ikh = 0; ikh < KH; ++ikh) {
                    ಪ್ರತಿ (int64_t ikw = 0; ikw < KW; ++ikw) {
                        ಧ್ರುವ int64_t iiw     = iow * s0 + ikw * d0 - p0;
                        ಧ್ರುವ int64_t iih     = ioh * s1 + ikh * d1 - p1;
                        ಧ್ರುವ int64_t dst_idx = iic * (KH * KW) + ikh * KW + ikw;

                        ಯದಿ (iiw < 0 || iiw >= IW || iih < 0 || iih >= IH) {
                            im2col_store_elem(dst_row, dst->type, dst_idx, 0.0f);
                        } ಅನ್ಯಥಾ {
                            ಧ್ರುವ int64_t src_idx = iih * IW + iiw;
                            ಧ್ರುವ ಭಿನ್ನ   ಮೌಲ್ಯ   = im2col_load_src_elem(src_channel, src1->type, src_idx);
                            im2col_store_elem(dst_row, dst->type, dst_idx, ಮೌಲ್ಯ);
                        }
                    }
                }
            }
        }
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}
