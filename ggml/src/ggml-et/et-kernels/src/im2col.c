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

स्थिर ओळीत शून्य im2col_store_elem(शून्य * dst_base, गणना ggml_type dst_type, int64_t idx, दशांश मूल्य) {
    जर (dst_type == GGML_TYPE_F32) {
        ((दशांश *) dst_base)[idx] = मूल्य;
    } अन्यथा {
        ((uint16_t *) dst_base)[idx] = fp32_to_fp16(मूल्य);
    }
}

स्थिर ओळीत दशांश im2col_load_src_elem(स्थिरांक शून्य * src_base, गणना ggml_type src_type, int64_t idx) {
    जर (src_type == GGML_TYPE_F32) {
        परत ((स्थिरांक दशांश *) src_base)[idx];
    }

    परत fp16_to_fp32(((स्थिरांक uint16_t *) src_base)[idx]);
}

पूर्णांक entry_point(संरचना ggml_et_द्विमान_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जर (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        परत -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    जर (thread_id < 0) {
        परत 0;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * src1 = &params->src1;
    संरचना ggml_tensor * dst  = &params->dst;

    जर (!src1->data || !dst->data) {
        परत -1;
    }

    जर (!((dst->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) ||
          (dst->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32)))) {
        परत -1;
    }

    स्थिरांक int32_t s0    = ((स्थिरांक int32_t *) dst->op_params)[0];
    स्थिरांक int32_t s1    = ((स्थिरांक int32_t *) dst->op_params)[1];
    स्थिरांक int32_t p0    = ((स्थिरांक int32_t *) dst->op_params)[2];
    स्थिरांक int32_t p1    = ((स्थिरांक int32_t *) dst->op_params)[3];
    स्थिरांक int32_t d0    = ((स्थिरांक int32_t *) dst->op_params)[4];
    स्थिरांक int32_t d1    = ((स्थिरांक int32_t *) dst->op_params)[5];
    स्थिरांक int32_t is_2d = ((स्थिरांक int32_t *) dst->op_params)[6];

    स्थिरांक int64_t N  = is_2d ? src1->ne[3] : src1->ne[2];
    स्थिरांक int64_t IC = is_2d ? src1->ne[2] : src1->ne[1];
    स्थिरांक int64_t IH = is_2d ? src1->ne[1] : 1;
    स्थिरांक int64_t IW = src1->ne[0];

    स्थिरांक int64_t KH = is_2d ? src0->ne[1] : 1;
    स्थिरांक int64_t KW = src0->ne[0];

    स्थिरांक int64_t OH         = is_2d ? dst->ne[2] : 1;
    स्थिरांक int64_t OW         = dst->ne[1];
    स्थिरांक int64_t row_elems  = dst->ne[0];
    स्थिरांक int64_t total_rows = OW * OH * N;

    स्थिरांक माप_प्रकार src_batch_stride   = is_2d ? src1->nb[3] : src1->nb[2];
    स्थिरांक माप_प्रकार src_channel_stride = is_2d ? src1->nb[2] : src1->nb[1];

    स्थिरांक माप_प्रकार dst_row_stride   = dst->nb[1];
    स्थिरांक माप_प्रकार dst_plane_stride = is_2d ? dst->nb[2] : 0;
    स्थिरांक माप_प्रकार dst_batch_stride = is_2d ? dst->nb[3] : dst->nb[2];

    स्थिरांक int64_t dst_elem_size  = (dst->type == GGML_TYPE_F32) ? (int64_t) आकार(दशांश) : (int64_t) आकार(uint16_t);
    स्थिरांक int64_t rows_per_group = et_rows_per_cacheline_group(row_elems, dst_elem_size);
    स्थिरांक int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

    साठी (int64_t grp = thread_id; grp < total_groups; grp += अंक_threads) {
        स्थिरांक int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        जर (row_end > total_rows) {
            row_end = total_rows;
        }

        साठी (int64_t row = row_start; row < row_end; ++row) {
            स्थिरांक int64_t in  = row / (OH * OW);
            स्थिरांक int64_t rem = row % (OH * OW);
            स्थिरांक int64_t ioh = rem / OW;
            स्थिरांक int64_t iow = rem % OW;

            शून्य * dst_row = (अक्षर *) dst->data + in * dst_batch_stride + ioh * dst_plane_stride + iow * dst_row_stride;

            साठी (int64_t iic = 0; iic < IC; ++iic) {
                स्थिरांक शून्य * src_channel = (स्थिरांक अक्षर *) src1->data + in * src_batch_stride + iic * src_channel_stride;

                साठी (int64_t ikh = 0; ikh < KH; ++ikh) {
                    साठी (int64_t ikw = 0; ikw < KW; ++ikw) {
                        स्थिरांक int64_t iiw     = iow * s0 + ikw * d0 - p0;
                        स्थिरांक int64_t iih     = ioh * s1 + ikh * d1 - p1;
                        स्थिरांक int64_t dst_idx = iic * (KH * KW) + ikh * KW + ikw;

                        जर (iiw < 0 || iiw >= IW || iih < 0 || iih >= IH) {
                            im2col_store_elem(dst_row, dst->type, dst_idx, 0.0f);
                        } अन्यथा {
                            स्थिरांक int64_t src_idx = iih * IW + iiw;
                            स्थिरांक दशांश   मूल्य   = im2col_load_src_elem(src_channel, src1->type, src_idx);
                            im2col_store_elem(dst_row, dst->type, dst_idx, मूल्य);
                        }
                    }
                }
            }
        }
    }

    परत 0;
}
