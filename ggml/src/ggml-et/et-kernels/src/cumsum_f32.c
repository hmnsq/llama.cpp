//******************************************************************************
// CUMSUM F32 Kernel
// Computes an inclusive prefix sum along dim 0 for each row in higher dims.
// First-pass implementation: scalar and row-contiguous input/output only.
//******************************************************************************

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <stdint.h>

கட்டமைப்பு ggml_et_cumsum_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor dst;
};

முழு entry_point(கட்டமைப்பு ggml_et_cumsum_params * params, வெறுமை * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    எனில் (!kernel_env) {
        திரும்பு -1;
    }

    முழு thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    முழு இலக்கம்_threads = get_இலக்கம்_threads(kernel_env->shire_mask);

    எனில் (thread_id < 0) {
        திரும்பு 0;
    }

    எனில் (params == 0 || ((uint64_t) params & 0x7) != 0) {
        திரும்பு -1;
    }

    கட்டமைப்பு ggml_tensor * src0 = &params->src0;
    கட்டமைப்பு ggml_tensor * dst  = &params->dst;

    எனில் (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        திரும்பு -1;
    }

    தசம * src0_data = (தசம *) src0->data;
    தசம * dst_data  = (தசம *) dst->data;

    எனில் (!src0_data || !dst_data) {
        திரும்பு -1;
    }

    மாறிலி int64_t ne0 = src0->ne[0];
    மாறிலி int64_t ne1 = src0->ne[1];
    மாறிலி int64_t ne2 = src0->ne[2];
    மாறிலி int64_t ne3 = src0->ne[3];

    மாறிலி size_t snb0 = src0->nb[0];
    மாறிலி size_t snb1 = src0->nb[1];
    மாறிலி size_t snb2 = src0->nb[2];
    மாறிலி size_t snb3 = src0->nb[3];

    மாறிலி size_t dnb0 = dst->nb[0];
    மாறிலி size_t dnb1 = dst->nb[1];
    மாறிலி size_t dnb2 = dst->nb[2];
    மாறிலி size_t dnb3 = dst->nb[3];

    எனில் (snb0 != அளவு(தசம) || dnb0 != அளவு(தசம)) {
        திரும்பு -1;
    }

    மாறிலி int64_t total_rows     = ne1 * ne2 * ne3;
    மாறிலி int64_t rows_per_group = et_rows_per_cacheline_group(ne0, அளவு(தசம));
    மாறிலி int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

    க்கு (int64_t grp = thread_id; grp < total_groups; grp += இலக்கம்_threads) {
        மாறிலி int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        எனில் (row_end > total_rows) {
            row_end = total_rows;
        }

        க்கு (int64_t row = row_start; row < row_end; ++row) {
            int64_t i1 = row % ne1;
            int64_t i2 = (row / ne1) % ne2;
            int64_t i3 = row / (ne1 * ne2);

            மாறிலி தசம * src_row = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0_data + i1 * snb1 + i2 * snb2 + i3 * snb3);
            தசம *       dst_row = (தசம *) ((எழுத்து *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            தசம acc = 0.0f;
            க்கு (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_row[i0];
                dst_row[i0] = acc;
            }
        }
    }

    திரும்பு 0;
}
