//******************************************************************************
// IM2COL Kernel
// Baseline scalar implementation for:
//   src1: [N, IC, IH, IW] -> dst: [N, OH, OW, IC*KH*KW] (2D)
//   src1: [N, IC,    IW] -> dst: [N,  1, OW, IC*   KW] (1D)
//
// Work is distributed by row-groups so threads own cache-line-aligned chunks of
// destination rows even when ne0 is not cache aligned.
//******************************************************************************

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "math_fp.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ im2col_store_elem(ଶୂନ୍ୟ * dst_base, ଗଣନା ggml_type dst_type, int64_t idx, ଦଶମିକ ମୂଲ୍ୟ) {
    ଯଦି (dst_type == GGML_TYPE_F32) {
        ((ଦଶମିକ *) dst_base)[idx] = ମୂଲ୍ୟ;
    } ଅନ୍ୟଥା {
        ((uint16_t *) dst_base)[idx] = fp32_to_fp16(ମୂଲ୍ୟ);
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ im2col_load_src_elem(ଅଚଳ ଶୂନ୍ୟ * src_base, ଗଣନା ggml_type src_type, int64_t idx) {
    ଯଦି (src_type == GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ((ଅଚଳ ଦଶମିକ *) src_base)[idx];
    }

    ଫେରନ୍ତୁ fp16_to_fp32(((ଅଚଳ uint16_t *) src_base)[idx]);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_ଦ୍ୱିମିକ_params * params, ଶୂନ୍ୟ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ଯଦି (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        ଫେରନ୍ତୁ -1;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_threads = get_ଅଙ୍କ_threads(kernel_env->shire_mask);

    ଯଦି (thread_id < 0) {
        ଫେରନ୍ତୁ 0;
    }

    ସଂରଚନା ggml_tensor * src0 = &params->src0;
    ସଂରଚନା ggml_tensor * src1 = &params->src1;
    ସଂରଚନା ggml_tensor * dst  = &params->dst;

    ଯଦି (!src1->data || !dst->data) {
        ଫେରନ୍ତୁ -1;
    }

    ଯଦି (!((dst->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) ||
          (dst->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32)))) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int32_t s0    = ((ଅଚଳ int32_t *) dst->op_params)[0];
    ଅଚଳ int32_t s1    = ((ଅଚଳ int32_t *) dst->op_params)[1];
    ଅଚଳ int32_t p0    = ((ଅଚଳ int32_t *) dst->op_params)[2];
    ଅଚଳ int32_t p1    = ((ଅଚଳ int32_t *) dst->op_params)[3];
    ଅଚଳ int32_t d0    = ((ଅଚଳ int32_t *) dst->op_params)[4];
    ଅଚଳ int32_t d1    = ((ଅଚଳ int32_t *) dst->op_params)[5];
    ଅଚଳ int32_t is_2d = ((ଅଚଳ int32_t *) dst->op_params)[6];

    ଅଚଳ int64_t N  = is_2d ? src1->ne[3] : src1->ne[2];
    ଅଚଳ int64_t IC = is_2d ? src1->ne[2] : src1->ne[1];
    ଅଚଳ int64_t IH = is_2d ? src1->ne[1] : 1;
    ଅଚଳ int64_t IW = src1->ne[0];

    ଅଚଳ int64_t KH = is_2d ? src0->ne[1] : 1;
    ଅଚଳ int64_t KW = src0->ne[0];

    ଅଚଳ int64_t OH         = is_2d ? dst->ne[2] : 1;
    ଅଚଳ int64_t OW         = dst->ne[1];
    ଅଚଳ int64_t row_elems  = dst->ne[0];
    ଅଚଳ int64_t total_rows = OW * OH * N;

    ଅଚଳ size_t src_batch_stride   = is_2d ? src1->nb[3] : src1->nb[2];
    ଅଚଳ size_t src_channel_stride = is_2d ? src1->nb[2] : src1->nb[1];

    ଅଚଳ size_t dst_row_stride   = dst->nb[1];
    ଅଚଳ size_t dst_plane_stride = is_2d ? dst->nb[2] : 0;
    ଅଚଳ size_t dst_batch_stride = is_2d ? dst->nb[3] : dst->nb[2];

    ଅଚଳ int64_t dst_elem_size  = (dst->type == GGML_TYPE_F32) ? (int64_t) ଆକାର(ଦଶମିକ) : (int64_t) ଆକାର(uint16_t);
    ଅଚଳ int64_t rows_per_group = et_rows_per_cacheline_group(row_elems, dst_elem_size);
    ଅଚଳ int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

    ପାଇଁ (int64_t grp = thread_id; grp < total_groups; grp += ଅଙ୍କ_threads) {
        ଅଚଳ int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        ଯଦି (row_end > total_rows) {
            row_end = total_rows;
        }

        ପାଇଁ (int64_t row = row_start; row < row_end; ++row) {
            ଅଚଳ int64_t in  = row / (OH * OW);
            ଅଚଳ int64_t rem = row % (OH * OW);
            ଅଚଳ int64_t ioh = rem / OW;
            ଅଚଳ int64_t iow = rem % OW;

            ଶୂନ୍ୟ * dst_row = (ଅକ୍ଷର *) dst->data + in * dst_batch_stride + ioh * dst_plane_stride + iow * dst_row_stride;

            ପାଇଁ (int64_t iic = 0; iic < IC; ++iic) {
                ଅଚଳ ଶୂନ୍ୟ * src_channel = (ଅଚଳ ଅକ୍ଷର *) src1->data + in * src_batch_stride + iic * src_channel_stride;

                ପାଇଁ (int64_t ikh = 0; ikh < KH; ++ikh) {
                    ପାଇଁ (int64_t ikw = 0; ikw < KW; ++ikw) {
                        ଅଚଳ int64_t iiw     = iow * s0 + ikw * d0 - p0;
                        ଅଚଳ int64_t iih     = ioh * s1 + ikh * d1 - p1;
                        ଅଚଳ int64_t dst_idx = iic * (KH * KW) + ikh * KW + ikw;

                        ଯଦି (iiw < 0 || iiw >= IW || iih < 0 || iih >= IH) {
                            im2col_store_elem(dst_row, dst->type, dst_idx, 0.0f);
                        } ଅନ୍ୟଥା {
                            ଅଚଳ int64_t src_idx = iih * IW + iiw;
                            ଅଚଳ ଦଶମିକ   ମୂଲ୍ୟ   = im2col_load_src_elem(src_channel, src1->type, src_idx);
                            im2col_store_elem(dst_row, dst->type, dst_idx, ମୂଲ୍ୟ);
                        }
                    }
                }
            }
        }
    }

    ଫେରନ୍ତୁ 0;
}
