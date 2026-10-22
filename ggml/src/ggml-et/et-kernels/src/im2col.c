//******************************************************************************
// IM2COL Kernel
// Baseline scalar implementation for:
//   src1: [N, IC, IH, IW] -> dst: [N, OH, OW, IC*KH*KW] (2D)
//   src1: [N, IC,    IW] -> dst: [N,  1, OW, IC*   KW] (1D)
//
// Work is distributed by row-groups so threads own cache-line-aligned chunks of
// destination rows even when ne0 is not cache aligned.
//******************************************************************************

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "math_fp.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <stdint.h>

நிலையான உள்வரி வெறுமை im2col_store_elem(வெறுமை * dst_base, எண்ணிக்கை ggml_type dst_type, int64_t idx, தசம மதிப்பு) {
    எனில் (dst_type == GGML_TYPE_F32) {
        ((தசம *) dst_base)[idx] = மதிப்பு;
    } இல்லையெனில் {
        ((uint16_t *) dst_base)[idx] = fp32_to_fp16(மதிப்பு);
    }
}

நிலையான உள்வரி தசம im2col_load_src_elem(மாறிலி வெறுமை * src_base, எண்ணிக்கை ggml_type src_type, int64_t idx) {
    எனில் (src_type == GGML_TYPE_F32) {
        திரும்பு ((மாறிலி தசம *) src_base)[idx];
    }

    திரும்பு fp16_to_fp32(((மாறிலி uint16_t *) src_base)[idx]);
}

முழு entry_point(கட்டமைப்பு ggml_et_இருமம்_params * params, வெறுமை * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    எனில் (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        திரும்பு -1;
    }

    முழு thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    முழு இலக்கம்_threads = get_இலக்கம்_threads(kernel_env->shire_mask);

    எனில் (thread_id < 0) {
        திரும்பு 0;
    }

    கட்டமைப்பு ggml_tensor * src0 = &params->src0;
    கட்டமைப்பு ggml_tensor * src1 = &params->src1;
    கட்டமைப்பு ggml_tensor * dst  = &params->dst;

    எனில் (!src1->data || !dst->data) {
        திரும்பு -1;
    }

    எனில் (!((dst->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) ||
          (dst->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32)))) {
        திரும்பு -1;
    }

    மாறிலி int32_t s0    = ((மாறிலி int32_t *) dst->op_params)[0];
    மாறிலி int32_t s1    = ((மாறிலி int32_t *) dst->op_params)[1];
    மாறிலி int32_t p0    = ((மாறிலி int32_t *) dst->op_params)[2];
    மாறிலி int32_t p1    = ((மாறிலி int32_t *) dst->op_params)[3];
    மாறிலி int32_t d0    = ((மாறிலி int32_t *) dst->op_params)[4];
    மாறிலி int32_t d1    = ((மாறிலி int32_t *) dst->op_params)[5];
    மாறிலி int32_t is_2d = ((மாறிலி int32_t *) dst->op_params)[6];

    மாறிலி int64_t N  = is_2d ? src1->ne[3] : src1->ne[2];
    மாறிலி int64_t IC = is_2d ? src1->ne[2] : src1->ne[1];
    மாறிலி int64_t IH = is_2d ? src1->ne[1] : 1;
    மாறிலி int64_t IW = src1->ne[0];

    மாறிலி int64_t KH = is_2d ? src0->ne[1] : 1;
    மாறிலி int64_t KW = src0->ne[0];

    மாறிலி int64_t OH         = is_2d ? dst->ne[2] : 1;
    மாறிலி int64_t OW         = dst->ne[1];
    மாறிலி int64_t row_elems  = dst->ne[0];
    மாறிலி int64_t total_rows = OW * OH * N;

    மாறிலி size_t src_batch_stride   = is_2d ? src1->nb[3] : src1->nb[2];
    மாறிலி size_t src_channel_stride = is_2d ? src1->nb[2] : src1->nb[1];

    மாறிலி size_t dst_row_stride   = dst->nb[1];
    மாறிலி size_t dst_plane_stride = is_2d ? dst->nb[2] : 0;
    மாறிலி size_t dst_batch_stride = is_2d ? dst->nb[3] : dst->nb[2];

    மாறிலி int64_t dst_elem_size  = (dst->type == GGML_TYPE_F32) ? (int64_t) அளவு(தசம) : (int64_t) அளவு(uint16_t);
    மாறிலி int64_t rows_per_group = et_rows_per_cacheline_group(row_elems, dst_elem_size);
    மாறிலி int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

    க்கு (int64_t grp = thread_id; grp < total_groups; grp += இலக்கம்_threads) {
        மாறிலி int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        எனில் (row_end > total_rows) {
            row_end = total_rows;
        }

        க்கு (int64_t row = row_start; row < row_end; ++row) {
            மாறிலி int64_t in  = row / (OH * OW);
            மாறிலி int64_t rem = row % (OH * OW);
            மாறிலி int64_t ioh = rem / OW;
            மாறிலி int64_t iow = rem % OW;

            வெறுமை * dst_row = (எழுத்து *) dst->data + in * dst_batch_stride + ioh * dst_plane_stride + iow * dst_row_stride;

            க்கு (int64_t iic = 0; iic < IC; ++iic) {
                மாறிலி வெறுமை * src_channel = (மாறிலி எழுத்து *) src1->data + in * src_batch_stride + iic * src_channel_stride;

                க்கு (int64_t ikh = 0; ikh < KH; ++ikh) {
                    க்கு (int64_t ikw = 0; ikw < KW; ++ikw) {
                        மாறிலி int64_t iiw     = iow * s0 + ikw * d0 - p0;
                        மாறிலி int64_t iih     = ioh * s1 + ikh * d1 - p1;
                        மாறிலி int64_t dst_idx = iic * (KH * KW) + ikh * KW + ikw;

                        எனில் (iiw < 0 || iiw >= IW || iih < 0 || iih >= IH) {
                            im2col_store_elem(dst_row, dst->type, dst_idx, 0.0f);
                        } இல்லையெனில் {
                            மாறிலி int64_t src_idx = iih * IW + iiw;
                            மாறிலி தசம   மதிப்பு   = im2col_load_src_elem(src_channel, src1->type, src_idx);
                            im2col_store_elem(dst_row, dst->type, dst_idx, மதிப்பு);
                        }
                    }
                }
            }
        }
    }

    திரும்பு 0;
}
