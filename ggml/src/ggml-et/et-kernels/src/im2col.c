//******************************************************************************
// IM2COL Kernel
// Baseline scalar implementation for:
//   src1: [N, IC, IH, IW] -> dst: [N, OH, OW, IC*KH*KW] (2D)
//   src1: [N, IC,    IW] -> dst: [N,  1, OW, IC*   KW] (1D)
//
// Work is distributed by row-groups so threads own cache-line-aligned chunks of
// destination rows even when ne0 is not cache aligned.
//******************************************************************************

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "math_fp.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

स्थिर अन्तःस्थ शून्य im2col_store_elem(शून्य * dst_base, गणना ggml_type dst_type, int64_t idx, भिन्न मानम्) {
    जइ (dst_type == GGML_TYPE_F32) {
        ((भिन्न *) dst_base)[idx] = मानम्;
    } अण्णहा {
        ((uint16_t *) dst_base)[idx] = fp32_to_fp16(मानम्);
    }
}

स्थिर अन्तःस्थ भिन्न im2col_load_src_elem(ध्रुव शून्य * src_base, गणना ggml_type src_type, int64_t idx) {
    जइ (src_type == GGML_TYPE_F32) {
        प्रत्यागम ((ध्रुव भिन्न *) src_base)[idx];
    }

    प्रत्यागम fp16_to_fp32(((ध्रुव uint16_t *) src_base)[idx]);
}

पूर्णाङ्क entry_point(संरचना ggml_et_द्विमान_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जइ (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अङ्कः_threads = get_अङ्कः_threads(kernel_env->shire_mask);

    जइ (thread_id < 0) {
        प्रत्यागम 0;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * src1 = &params->src1;
    संरचना ggml_tensor * dst  = &params->dst;

    जइ (!src1->data || !dst->data) {
        प्रत्यागम -1;
    }

    जइ (!((dst->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) ||
          (dst->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32)))) {
        प्रत्यागम -1;
    }

    ध्रुव int32_t s0    = ((ध्रुव int32_t *) dst->op_params)[0];
    ध्रुव int32_t s1    = ((ध्रुव int32_t *) dst->op_params)[1];
    ध्रुव int32_t p0    = ((ध्रुव int32_t *) dst->op_params)[2];
    ध्रुव int32_t p1    = ((ध्रुव int32_t *) dst->op_params)[3];
    ध्रुव int32_t d0    = ((ध्रुव int32_t *) dst->op_params)[4];
    ध्रुव int32_t d1    = ((ध्रुव int32_t *) dst->op_params)[5];
    ध्रुव int32_t is_2d = ((ध्रुव int32_t *) dst->op_params)[6];

    ध्रुव int64_t N  = is_2d ? src1->ne[3] : src1->ne[2];
    ध्रुव int64_t IC = is_2d ? src1->ne[2] : src1->ne[1];
    ध्रुव int64_t IH = is_2d ? src1->ne[1] : 1;
    ध्रुव int64_t IW = src1->ne[0];

    ध्रुव int64_t KH = is_2d ? src0->ne[1] : 1;
    ध्रुव int64_t KW = src0->ne[0];

    ध्रुव int64_t OH         = is_2d ? dst->ne[2] : 1;
    ध्रुव int64_t OW         = dst->ne[1];
    ध्रुव int64_t row_elems  = dst->ne[0];
    ध्रुव int64_t total_rows = OW * OH * N;

    ध्रुव size_t src_batch_stride   = is_2d ? src1->nb[3] : src1->nb[2];
    ध्रुव size_t src_channel_stride = is_2d ? src1->nb[2] : src1->nb[1];

    ध्रुव size_t dst_row_stride   = dst->nb[1];
    ध्रुव size_t dst_plane_stride = is_2d ? dst->nb[2] : 0;
    ध्रुव size_t dst_batch_stride = is_2d ? dst->nb[3] : dst->nb[2];

    ध्रुव int64_t dst_elem_size  = (dst->type == GGML_TYPE_F32) ? (int64_t) परिमाण(भिन्न) : (int64_t) परिमाण(uint16_t);
    ध्रुव int64_t rows_per_group = et_rows_per_cacheline_group(row_elems, dst_elem_size);
    ध्रुव int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

    प्रति (int64_t grp = thread_id; grp < total_groups; grp += अङ्कः_threads) {
        ध्रुव int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        जइ (row_end > total_rows) {
            row_end = total_rows;
        }

        प्रति (int64_t row = row_start; row < row_end; ++row) {
            ध्रुव int64_t in  = row / (OH * OW);
            ध्रुव int64_t rem = row % (OH * OW);
            ध्रुव int64_t ioh = rem / OW;
            ध्रुव int64_t iow = rem % OW;

            शून्य * dst_row = (वर्ण *) dst->data + in * dst_batch_stride + ioh * dst_plane_stride + iow * dst_row_stride;

            प्रति (int64_t iic = 0; iic < IC; ++iic) {
                ध्रुव शून्य * src_channel = (ध्रुव वर्ण *) src1->data + in * src_batch_stride + iic * src_channel_stride;

                प्रति (int64_t ikh = 0; ikh < KH; ++ikh) {
                    प्रति (int64_t ikw = 0; ikw < KW; ++ikw) {
                        ध्रुव int64_t iiw     = iow * s0 + ikw * d0 - p0;
                        ध्रुव int64_t iih     = ioh * s1 + ikh * d1 - p1;
                        ध्रुव int64_t dst_idx = iic * (KH * KW) + ikh * KW + ikw;

                        जइ (iiw < 0 || iiw >= IW || iih < 0 || iih >= IH) {
                            im2col_store_elem(dst_row, dst->type, dst_idx, 0.0f);
                        } अण्णहा {
                            ध्रुव int64_t src_idx = iih * IW + iiw;
                            ध्रुव भिन्न   मानम्   = im2col_load_src_elem(src_channel, src1->type, src_idx);
                            im2col_store_elem(dst_row, dst->type, dst_idx, मानम्);
                        }
                    }
                }
            }
        }
    }

    प्रत्यागम 0;
}
