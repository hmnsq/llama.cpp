//******************************************************************************
// IM2COL Kernel
// Baseline scalar implementation for:
//   src1: [N, IC, IH, IW] -> dst: [N, OH, OW, IC*KH*KW] (2D)
//   src1: [N, IC,    IW] -> dst: [N,  1, OW, IC*   KW] (1D)
//
// Work is distributed by row-groups so threads own cache-line-aligned chunks of
// destination rows even when ne0 is not cache aligned.
//******************************************************************************

#समावेश "ggml_tensor.h"
#समावेश "math_fp.h"
#समावेश "platform.h"

#समावेश <stdint.h>

अटल अंतरभूत व्योम im2col_store_elem(व्योम * dst_base, क्रमागत ggml_type dst_type, int64_t idx, भग्न मूल्य) {
    अगर (dst_type == GGML_TYPE_F32) {
        ((भग्न *) dst_base)[idx] = मूल्य;
    } अन्यथा {
        ((uint16_t *) dst_base)[idx] = fp32_to_fp16(मूल्य);
    }
}

अटल अंतरभूत भग्न im2col_load_src_elem(स्थिर व्योम * src_base, क्रमागत ggml_type src_type, int64_t idx) {
    अगर (src_type == GGML_TYPE_F32) {
        वापस ((स्थिर भग्न *) src_base)[idx];
    }

    वापस fp16_to_fp32(((स्थिर uint16_t *) src_base)[idx]);
}

पूर्णांक entry_point(काष्ठा ggml_et_द्विआधारी_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस 0;
    }

    काष्ठा ggml_tensor * src0 = &params->src0;
    काष्ठा ggml_tensor * src1 = &params->src1;
    काष्ठा ggml_tensor * dst  = &params->dst;

    अगर (!src1->तथ्य || !dst->तथ्य) {
        वापस -1;
    }

    अगर (!((dst->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) ||
          (dst->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32)))) {
        वापस -1;
    }

    स्थिर int32_t s0    = ((स्थिर int32_t *) dst->op_params)[0];
    स्थिर int32_t s1    = ((स्थिर int32_t *) dst->op_params)[1];
    स्थिर int32_t p0    = ((स्थिर int32_t *) dst->op_params)[2];
    स्थिर int32_t p1    = ((स्थिर int32_t *) dst->op_params)[3];
    स्थिर int32_t d0    = ((स्थिर int32_t *) dst->op_params)[4];
    स्थिर int32_t d1    = ((स्थिर int32_t *) dst->op_params)[5];
    स्थिर int32_t is_2d = ((स्थिर int32_t *) dst->op_params)[6];

    स्थिर int64_t N  = is_2d ? src1->ne[3] : src1->ne[2];
    स्थिर int64_t IC = is_2d ? src1->ne[2] : src1->ne[1];
    स्थिर int64_t IH = is_2d ? src1->ne[1] : 1;
    स्थिर int64_t IW = src1->ne[0];

    स्थिर int64_t KH = is_2d ? src0->ne[1] : 1;
    स्थिर int64_t KW = src0->ne[0];

    स्थिर int64_t OH         = is_2d ? dst->ne[2] : 1;
    स्थिर int64_t OW         = dst->ne[1];
    स्थिर int64_t row_elems  = dst->ne[0];
    स्थिर int64_t योग_rows = OW * OH * N;

    स्थिर माप_प्रकार src_batch_stride   = is_2d ? src1->nb[3] : src1->nb[2];
    स्थिर माप_प्रकार src_channel_stride = is_2d ? src1->nb[2] : src1->nb[1];

    स्थिर माप_प्रकार dst_row_stride   = dst->nb[1];
    स्थिर माप_प्रकार dst_plane_stride = is_2d ? dst->nb[2] : 0;
    स्थिर माप_प्रकार dst_batch_stride = is_2d ? dst->nb[3] : dst->nb[2];

    स्थिर int64_t dst_elem_आकार  = (dst->type == GGML_TYPE_F32) ? (int64_t) माप(भग्न) : (int64_t) माप(uint16_t);
    स्थिर int64_t rows_per_group = et_rows_per_cacheline_group(row_elems, dst_elem_आकार);
    स्थिर int64_t योग_groups   = (योग_rows + rows_per_group - 1) / rows_per_group;

    क्रम (int64_t grp = thread_id; grp < योग_groups; grp += अंक_threads) {
        स्थिर int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        अगर (row_end > योग_rows) {
            row_end = योग_rows;
        }

        क्रम (int64_t row = row_start; row < row_end; ++row) {
            स्थिर int64_t in  = row / (OH * OW);
            स्थिर int64_t rem = row % (OH * OW);
            स्थिर int64_t ioh = rem / OW;
            स्थिर int64_t iow = rem % OW;

            व्योम * dst_row = (अक्षर *) dst->तथ्य + in * dst_batch_stride + ioh * dst_plane_stride + iow * dst_row_stride;

            क्रम (int64_t iic = 0; iic < IC; ++iic) {
                स्थिर व्योम * src_channel = (स्थिर अक्षर *) src1->तथ्य + in * src_batch_stride + iic * src_channel_stride;

                क्रम (int64_t ikh = 0; ikh < KH; ++ikh) {
                    क्रम (int64_t ikw = 0; ikw < KW; ++ikw) {
                        स्थिर int64_t iiw     = iow * s0 + ikw * d0 - p0;
                        स्थिर int64_t iih     = ioh * s1 + ikh * d1 - p1;
                        स्थिर int64_t dst_idx = iic * (KH * KW) + ikh * KW + ikw;

                        अगर (iiw < 0 || iiw >= IW || iih < 0 || iih >= IH) {
                            im2col_store_elem(dst_row, dst->type, dst_idx, 0.0f);
                        } अन्यथा {
                            स्थिर int64_t src_idx = iih * IW + iiw;
                            स्थिर भग्न   मूल्य   = im2col_load_src_elem(src_channel, src1->type, src_idx);
                            im2col_store_elem(dst_row, dst->type, dst_idx, मूल्य);
                        }
                    }
                }
            }
        }
    }

    वापस 0;
}
