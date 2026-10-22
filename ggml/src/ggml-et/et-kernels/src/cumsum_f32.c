//******************************************************************************
// CUMSUM F32 Kernel
// Computes an inclusive prefix sum along dim 0 for each row in higher dims.
// First-pass implementation: scalar and row-contiguous input/output only.
//******************************************************************************

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>

ସଂରଚନା ggml_et_cumsum_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor dst;
};

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_cumsum_params * params, ଶୂନ୍ୟ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ଯଦି (!kernel_env) {
        ଫେରନ୍ତୁ -1;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_threads = get_ଅଙ୍କ_threads(kernel_env->shire_mask);

    ଯଦି (thread_id < 0) {
        ଫେରନ୍ତୁ 0;
    }

    ଯଦି (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ଫେରନ୍ତୁ -1;
    }

    ସଂରଚନା ggml_tensor * src0 = &params->src0;
    ସଂରଚନା ggml_tensor * dst  = &params->dst;

    ଯଦି (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ -1;
    }

    ଦଶମିକ * src0_data = (ଦଶମିକ *) src0->data;
    ଦଶମିକ * dst_data  = (ଦଶମିକ *) dst->data;

    ଯଦି (!src0_data || !dst_data) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int64_t ne0 = src0->ne[0];
    ଅଚଳ int64_t ne1 = src0->ne[1];
    ଅଚଳ int64_t ne2 = src0->ne[2];
    ଅଚଳ int64_t ne3 = src0->ne[3];

    ଅଚଳ size_t snb0 = src0->nb[0];
    ଅଚଳ size_t snb1 = src0->nb[1];
    ଅଚଳ size_t snb2 = src0->nb[2];
    ଅଚଳ size_t snb3 = src0->nb[3];

    ଅଚଳ size_t dnb0 = dst->nb[0];
    ଅଚଳ size_t dnb1 = dst->nb[1];
    ଅଚଳ size_t dnb2 = dst->nb[2];
    ଅଚଳ size_t dnb3 = dst->nb[3];

    ଯଦି (snb0 != ଆକାର(ଦଶମିକ) || dnb0 != ଆକାର(ଦଶମିକ)) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int64_t total_rows     = ne1 * ne2 * ne3;
    ଅଚଳ int64_t rows_per_group = et_rows_per_cacheline_group(ne0, ଆକାର(ଦଶମିକ));
    ଅଚଳ int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

    ପାଇଁ (int64_t grp = thread_id; grp < total_groups; grp += ଅଙ୍କ_threads) {
        ଅଚଳ int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        ଯଦି (row_end > total_rows) {
            row_end = total_rows;
        }

        ପାଇଁ (int64_t row = row_start; row < row_end; ++row) {
            int64_t i1 = row % ne1;
            int64_t i2 = (row / ne1) % ne2;
            int64_t i3 = row / (ne1 * ne2);

            ଅଚଳ ଦଶମିକ * src_row = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0_data + i1 * snb1 + i2 * snb2 + i3 * snb3);
            ଦଶମିକ *       dst_row = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            ଦଶମିକ acc = 0.0f;
            ପାଇଁ (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_row[i0];
                dst_row[i0] = acc;
            }
        }
    }

    ଫେରନ୍ତୁ 0;
}
