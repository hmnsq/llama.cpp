//******************************************************************************
// IM2COL Kernel
// Baseline scalar implementation for:
//   src1: [N, IC, IH, IW] -> dst: [N, OH, OW, IC*KH*KW] (2D)
//   src1: [N, IC,    IW] -> dst: [N,  1, OW, IC*   KW] (1D)
//
// Work is distributed by row-groups so threads own cache-line-aligned chunks of
// destination rows even when ne0 is not cache aligned.
//******************************************************************************

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "math_fp.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

স্থির অন্তঃরেখ শূন্য im2col_store_elem(শূন্য * dst_base, গণনা ggml_type dst_type, int64_t idx, দশমিক মান) {
    যদি (dst_type == GGML_TYPE_F32) {
        ((দশমিক *) dst_base)[idx] = মান;
    } অন্যথা {
        ((uint16_t *) dst_base)[idx] = fp32_to_fp16(মান);
    }
}

স্থির অন্তঃরেখ দশমিক im2col_load_src_elem(ধ্রুবক শূন্য * src_base, গণনা ggml_type src_type, int64_t idx) {
    যদি (src_type == GGML_TYPE_F32) {
        ফেরত ((ধ্রুবক দশমিক *) src_base)[idx];
    }

    ফেরত fp16_to_fp32(((ধ্রুবক uint16_t *) src_base)[idx]);
}

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_দ্বিমিক_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        ফেরত -1;
    }

    পূর্ণসংখ্যা thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণসংখ্যা অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ফেরত 0;
    }

    কাঠামো ggml_tensor * src0 = &params->src0;
    কাঠামো ggml_tensor * src1 = &params->src1;
    কাঠামো ggml_tensor * dst  = &params->dst;

    যদি (!src1->তথ্য || !dst->তথ্য) {
        ফেরত -1;
    }

    যদি (!((dst->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) ||
          (dst->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32)))) {
        ফেরত -1;
    }

    ধ্রুবক int32_t s0    = ((ধ্রুবক int32_t *) dst->op_params)[0];
    ধ্রুবক int32_t s1    = ((ধ্রুবক int32_t *) dst->op_params)[1];
    ধ্রুবক int32_t p0    = ((ধ্রুবক int32_t *) dst->op_params)[2];
    ধ্রুবক int32_t p1    = ((ধ্রুবক int32_t *) dst->op_params)[3];
    ধ্রুবক int32_t d0    = ((ধ্রুবক int32_t *) dst->op_params)[4];
    ধ্রুবক int32_t d1    = ((ধ্রুবক int32_t *) dst->op_params)[5];
    ধ্রুবক int32_t is_2d = ((ধ্রুবক int32_t *) dst->op_params)[6];

    ধ্রুবক int64_t N  = is_2d ? src1->ne[3] : src1->ne[2];
    ধ্রুবক int64_t IC = is_2d ? src1->ne[2] : src1->ne[1];
    ধ্রুবক int64_t IH = is_2d ? src1->ne[1] : 1;
    ধ্রুবক int64_t IW = src1->ne[0];

    ধ্রুবক int64_t KH = is_2d ? src0->ne[1] : 1;
    ধ্রুবক int64_t KW = src0->ne[0];

    ধ্রুবক int64_t OH         = is_2d ? dst->ne[2] : 1;
    ধ্রুবক int64_t OW         = dst->ne[1];
    ধ্রুবক int64_t row_elems  = dst->ne[0];
    ধ্রুবক int64_t যোগ_rows = OW * OH * N;

    ধ্রুবক size_t src_batch_stride   = is_2d ? src1->nb[3] : src1->nb[2];
    ধ্রুবক size_t src_channel_stride = is_2d ? src1->nb[2] : src1->nb[1];

    ধ্রুবক size_t dst_row_stride   = dst->nb[1];
    ধ্রুবক size_t dst_plane_stride = is_2d ? dst->nb[2] : 0;
    ধ্রুবক size_t dst_batch_stride = is_2d ? dst->nb[3] : dst->nb[2];

    ধ্রুবক int64_t dst_elem_আকার  = (dst->type == GGML_TYPE_F32) ? (int64_t) আকার(দশমিক) : (int64_t) আকার(uint16_t);
    ধ্রুবক int64_t rows_per_group = et_rows_per_cacheline_group(row_elems, dst_elem_আকার);
    ধ্রুবক int64_t যোগ_groups   = (যোগ_rows + rows_per_group - 1) / rows_per_group;

    জন্য (int64_t grp = thread_id; grp < যোগ_groups; grp += অঙ্ক_threads) {
        ধ্রুবক int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        যদি (row_end > যোগ_rows) {
            row_end = যোগ_rows;
        }

        জন্য (int64_t row = row_start; row < row_end; ++row) {
            ধ্রুবক int64_t in  = row / (OH * OW);
            ধ্রুবক int64_t rem = row % (OH * OW);
            ধ্রুবক int64_t ioh = rem / OW;
            ধ্রুবক int64_t iow = rem % OW;

            শূন্য * dst_row = (অক্ষর *) dst->তথ্য + in * dst_batch_stride + ioh * dst_plane_stride + iow * dst_row_stride;

            জন্য (int64_t iic = 0; iic < IC; ++iic) {
                ধ্রুবক শূন্য * src_channel = (ধ্রুবক অক্ষর *) src1->তথ্য + in * src_batch_stride + iic * src_channel_stride;

                জন্য (int64_t ikh = 0; ikh < KH; ++ikh) {
                    জন্য (int64_t ikw = 0; ikw < KW; ++ikw) {
                        ধ্রুবক int64_t iiw     = iow * s0 + ikw * d0 - p0;
                        ধ্রুবক int64_t iih     = ioh * s1 + ikh * d1 - p1;
                        ধ্রুবক int64_t dst_idx = iic * (KH * KW) + ikh * KW + ikw;

                        যদি (iiw < 0 || iiw >= IW || iih < 0 || iih >= IH) {
                            im2col_store_elem(dst_row, dst->type, dst_idx, 0.0f);
                        } অন্যথা {
                            ধ্রুবক int64_t src_idx = iih * IW + iiw;
                            ধ্রুবক দশমিক   মান   = im2col_load_src_elem(src_channel, src1->type, src_idx);
                            im2col_store_elem(dst_row, dst->type, dst_idx, মান);
                        }
                    }
                }
            }
        }
    }

    ফেরত 0;
}
