//******************************************************************************
// IM2COL Kernel
// Baseline scalar implementation for:
//   src1: [N, IC, IH, IW] -> dst: [N, OH, OW, IC*KH*KW] (2D)
//   src1: [N, IC,    IW] -> dst: [N,  1, OW, IC*   KW] (1D)
//
// Work is distributed by row-groups so threads own cache-line-aligned chunks of
// destination rows even when ne0 is not cache aligned.
//******************************************************************************

#অন্তর্ভাৱ "ggml_tensor.h"
#অন্তর্ভাৱ "math_fp.h"
#অন্তর্ভাৱ "platform.h"

#অন্তর্ভাৱ <stdint.h>

স্থির অন্তঃস্থ শূন্য im2col_store_elem(শূন্য * dst_base, গণনা ggml_type dst_type, int64_t idx, ভিন্ন মান) {
    যদি (dst_type == GGML_TYPE_F32) {
        ((ভিন্ন *) dst_base)[idx] = মান;
    } অন্যথা {
        ((uint16_t *) dst_base)[idx] = fp32_to_fp16(মান);
    }
}

স্থির অন্তঃস্থ ভিন্ন im2col_load_src_elem(ধ্রুৱ শূন্য * src_base, গণনা ggml_type src_type, int64_t idx) {
    যদি (src_type == GGML_TYPE_F32) {
        ঘূৰাই ((ধ্রুৱ ভিন্ন *) src_base)[idx];
    }

    ঘূৰাই fp16_to_fp32(((ধ্রুৱ uint16_t *) src_base)[idx]);
}

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_দ্বিমিক_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        ঘূৰাই -1;
    }

    পূর্ণাঙ্ক thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণাঙ্ক অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ঘূৰাই 0;
    }

    সংরচনা ggml_tensor * src0 = &params->src0;
    সংরচনা ggml_tensor * src1 = &params->src1;
    সংরচনা ggml_tensor * dst  = &params->dst;

    যদি (!src1->data || !dst->data) {
        ঘূৰাই -1;
    }

    যদি (!((dst->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) ||
          (dst->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32)))) {
        ঘূৰাই -1;
    }

    ধ্রুৱ int32_t s0    = ((ধ্রুৱ int32_t *) dst->op_params)[0];
    ধ্রুৱ int32_t s1    = ((ধ্রুৱ int32_t *) dst->op_params)[1];
    ধ্রুৱ int32_t p0    = ((ধ্রুৱ int32_t *) dst->op_params)[2];
    ধ্রুৱ int32_t p1    = ((ধ্রুৱ int32_t *) dst->op_params)[3];
    ধ্রুৱ int32_t d0    = ((ধ্রুৱ int32_t *) dst->op_params)[4];
    ধ্রুৱ int32_t d1    = ((ধ্রুৱ int32_t *) dst->op_params)[5];
    ধ্রুৱ int32_t is_2d = ((ধ্রুৱ int32_t *) dst->op_params)[6];

    ধ্রুৱ int64_t N  = is_2d ? src1->ne[3] : src1->ne[2];
    ধ্রুৱ int64_t IC = is_2d ? src1->ne[2] : src1->ne[1];
    ধ্রুৱ int64_t IH = is_2d ? src1->ne[1] : 1;
    ধ্রুৱ int64_t IW = src1->ne[0];

    ধ্রুৱ int64_t KH = is_2d ? src0->ne[1] : 1;
    ধ্রুৱ int64_t KW = src0->ne[0];

    ধ্রুৱ int64_t OH         = is_2d ? dst->ne[2] : 1;
    ধ্রুৱ int64_t OW         = dst->ne[1];
    ধ্রুৱ int64_t row_elems  = dst->ne[0];
    ধ্রুৱ int64_t total_rows = OW * OH * N;

    ধ্রুৱ size_t src_batch_stride   = is_2d ? src1->nb[3] : src1->nb[2];
    ধ্রুৱ size_t src_channel_stride = is_2d ? src1->nb[2] : src1->nb[1];

    ধ্রুৱ size_t dst_row_stride   = dst->nb[1];
    ধ্রুৱ size_t dst_plane_stride = is_2d ? dst->nb[2] : 0;
    ধ্রুৱ size_t dst_batch_stride = is_2d ? dst->nb[3] : dst->nb[2];

    ধ্রুৱ int64_t dst_elem_size  = (dst->type == GGML_TYPE_F32) ? (int64_t) পরিমাণ(ভিন্ন) : (int64_t) পরিমাণ(uint16_t);
    ধ্রুৱ int64_t rows_per_group = et_rows_per_cacheline_group(row_elems, dst_elem_size);
    ধ্রুৱ int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

    বাবে (int64_t grp = thread_id; grp < total_groups; grp += অঙ্ক_threads) {
        ধ্রুৱ int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        যদি (row_end > total_rows) {
            row_end = total_rows;
        }

        বাবে (int64_t row = row_start; row < row_end; ++row) {
            ধ্রুৱ int64_t in  = row / (OH * OW);
            ধ্রুৱ int64_t rem = row % (OH * OW);
            ধ্রুৱ int64_t ioh = rem / OW;
            ধ্রুৱ int64_t iow = rem % OW;

            শূন্য * dst_row = (ৱর্ণ *) dst->data + in * dst_batch_stride + ioh * dst_plane_stride + iow * dst_row_stride;

            বাবে (int64_t iic = 0; iic < IC; ++iic) {
                ধ্রুৱ শূন্য * src_channel = (ধ্রুৱ ৱর্ণ *) src1->data + in * src_batch_stride + iic * src_channel_stride;

                বাবে (int64_t ikh = 0; ikh < KH; ++ikh) {
                    বাবে (int64_t ikw = 0; ikw < KW; ++ikw) {
                        ধ্রুৱ int64_t iiw     = iow * s0 + ikw * d0 - p0;
                        ধ্রুৱ int64_t iih     = ioh * s1 + ikh * d1 - p1;
                        ধ্রুৱ int64_t dst_idx = iic * (KH * KW) + ikh * KW + ikw;

                        যদি (iiw < 0 || iiw >= IW || iih < 0 || iih >= IH) {
                            im2col_store_elem(dst_row, dst->type, dst_idx, 0.0f);
                        } অন্যথা {
                            ধ্রুৱ int64_t src_idx = iih * IW + iiw;
                            ধ্রুৱ ভিন্ন   মান   = im2col_load_src_elem(src_channel, src1->type, src_idx);
                            im2col_store_elem(dst_row, dst->type, dst_idx, মান);
                        }
                    }
                }
            }
        }
    }

    ঘূৰাই 0;
}
