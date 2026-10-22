//******************************************************************************
// CUMSUM F32 Kernel
// Computes an inclusive prefix sum along dim 0 for each row in higher dims.
// First-pass implementation: scalar and row-contiguous input/output only.
//******************************************************************************

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

संरचना ggml_et_cumsum_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor dst;
};

पूर्णांक entry_point(संरचना ggml_et_cumsum_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जर (!kernel_env) {
        परत -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    जर (thread_id < 0) {
        परत 0;
    }

    जर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        परत -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    जर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        परत -1;
    }

    दशांश * src0_data = (दशांश *) src0->data;
    दशांश * dst_data  = (दशांश *) dst->data;

    जर (!src0_data || !dst_data) {
        परत -1;
    }

    स्थिरांक int64_t ne0 = src0->ne[0];
    स्थिरांक int64_t ne1 = src0->ne[1];
    स्थिरांक int64_t ne2 = src0->ne[2];
    स्थिरांक int64_t ne3 = src0->ne[3];

    स्थिरांक माप_प्रकार snb0 = src0->nb[0];
    स्थिरांक माप_प्रकार snb1 = src0->nb[1];
    स्थिरांक माप_प्रकार snb2 = src0->nb[2];
    स्थिरांक माप_प्रकार snb3 = src0->nb[3];

    स्थिरांक माप_प्रकार dnb0 = dst->nb[0];
    स्थिरांक माप_प्रकार dnb1 = dst->nb[1];
    स्थिरांक माप_प्रकार dnb2 = dst->nb[2];
    स्थिरांक माप_प्रकार dnb3 = dst->nb[3];

    जर (snb0 != आकार(दशांश) || dnb0 != आकार(दशांश)) {
        परत -1;
    }

    स्थिरांक int64_t total_rows     = ne1 * ne2 * ne3;
    स्थिरांक int64_t rows_per_group = et_rows_per_cacheline_group(ne0, आकार(दशांश));
    स्थिरांक int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

    साठी (int64_t grp = thread_id; grp < total_groups; grp += अंक_threads) {
        स्थिरांक int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        जर (row_end > total_rows) {
            row_end = total_rows;
        }

        साठी (int64_t row = row_start; row < row_end; ++row) {
            int64_t i1 = row % ne1;
            int64_t i2 = (row / ne1) % ne2;
            int64_t i3 = row / (ne1 * ne2);

            स्थिरांक दशांश * src_row = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src0_data + i1 * snb1 + i2 * snb2 + i3 * snb3);
            दशांश *       dst_row = (दशांश *) ((अक्षर *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            दशांश acc = 0.0f;
            साठी (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_row[i0];
                dst_row[i0] = acc;
            }
        }
    }

    परत 0;
}
