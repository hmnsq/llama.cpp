//******************************************************************************
// CUMSUM F32 Kernel
// Computes an inclusive prefix sum along dim 0 for each row in higher dims.
// First-pass implementation: scalar and row-contiguous input/output only.
//******************************************************************************

#समावेश "ggml_tensor.h"
#समावेश "platform.h"

#समावेश <stdint.h>

काष्ठा ggml_et_cumsum_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor dst;
};

पूर्णांक entry_point(काष्ठा ggml_et_cumsum_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस 0;
    }

    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;
    }

    काष्ठा ggml_tensor * src0 = &params->src0;
    काष्ठा ggml_tensor * dst  = &params->dst;

    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस -1;
    }

    भग्न * src0_तथ्य = (भग्न *) src0->तथ्य;
    भग्न * dst_तथ्य  = (भग्न *) dst->तथ्य;

    अगर (!src0_तथ्य || !dst_तथ्य) {
        वापस -1;
    }

    स्थिर int64_t ne0 = src0->ne[0];
    स्थिर int64_t ne1 = src0->ne[1];
    स्थिर int64_t ne2 = src0->ne[2];
    स्थिर int64_t ne3 = src0->ne[3];

    स्थिर माप_प्रकार snb0 = src0->nb[0];
    स्थिर माप_प्रकार snb1 = src0->nb[1];
    स्थिर माप_प्रकार snb2 = src0->nb[2];
    स्थिर माप_प्रकार snb3 = src0->nb[3];

    स्थिर माप_प्रकार dnb0 = dst->nb[0];
    स्थिर माप_प्रकार dnb1 = dst->nb[1];
    स्थिर माप_प्रकार dnb2 = dst->nb[2];
    स्थिर माप_प्रकार dnb3 = dst->nb[3];

    अगर (snb0 != माप(भग्न) || dnb0 != माप(भग्न)) {
        वापस -1;
    }

    स्थिर int64_t योग_rows     = ne1 * ne2 * ne3;
    स्थिर int64_t rows_per_group = et_rows_per_cacheline_group(ne0, माप(भग्न));
    स्थिर int64_t योग_groups   = (योग_rows + rows_per_group - 1) / rows_per_group;

    क्रम (int64_t grp = thread_id; grp < योग_groups; grp += अंक_threads) {
        स्थिर int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        अगर (row_end > योग_rows) {
            row_end = योग_rows;
        }

        क्रम (int64_t row = row_start; row < row_end; ++row) {
            int64_t i1 = row % ne1;
            int64_t i2 = (row / ne1) % ne2;
            int64_t i3 = row / (ne1 * ne2);

            स्थिर भग्न * src_row = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i1 * snb1 + i2 * snb2 + i3 * snb3);
            भग्न *       dst_row = (भग्न *) ((अक्षर *) dst_तथ्य + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            भग्न acc = 0.0f;
            क्रम (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_row[i0];
                dst_row[i0] = acc;
            }
        }
    }

    वापस 0;
}
