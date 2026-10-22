//******************************************************************************
// CUMSUM F32 Kernel
// Computes an inclusive prefix sum along dim 0 for each row in higher dims.
// First-pass implementation: scalar and row-contiguous input/output only.
//******************************************************************************

#অন্তর্ভাৱ "ggml_tensor.h"
#অন্তর্ভাৱ "platform.h"

#অন্তর্ভাৱ <stdint.h>

সংরচনা ggml_et_cumsum_params {
    সংরচনা ggml_tensor src0;
    সংরচনা ggml_tensor dst;
};

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_cumsum_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ঘূৰাই -1;
    }

    পূর্ণাঙ্ক thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণাঙ্ক অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ঘূৰাই 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ঘূৰাই -1;
    }

    সংরচনা ggml_tensor * src0 = &params->src0;
    সংরচনা ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ঘূৰাই -1;
    }

    ভিন্ন * src0_data = (ভিন্ন *) src0->data;
    ভিন্ন * dst_data  = (ভিন্ন *) dst->data;

    যদি (!src0_data || !dst_data) {
        ঘূৰাই -1;
    }

    ধ্রুৱ int64_t ne0 = src0->ne[0];
    ধ্রুৱ int64_t ne1 = src0->ne[1];
    ধ্রুৱ int64_t ne2 = src0->ne[2];
    ধ্রুৱ int64_t ne3 = src0->ne[3];

    ধ্রুৱ size_t snb0 = src0->nb[0];
    ধ্রুৱ size_t snb1 = src0->nb[1];
    ধ্রুৱ size_t snb2 = src0->nb[2];
    ধ্রুৱ size_t snb3 = src0->nb[3];

    ধ্রুৱ size_t dnb0 = dst->nb[0];
    ধ্রুৱ size_t dnb1 = dst->nb[1];
    ধ্রুৱ size_t dnb2 = dst->nb[2];
    ধ্রুৱ size_t dnb3 = dst->nb[3];

    যদি (snb0 != পরিমাণ(ভিন্ন) || dnb0 != পরিমাণ(ভিন্ন)) {
        ঘূৰাই -1;
    }

    ধ্রুৱ int64_t total_rows     = ne1 * ne2 * ne3;
    ধ্রুৱ int64_t rows_per_group = et_rows_per_cacheline_group(ne0, পরিমাণ(ভিন্ন));
    ধ্রুৱ int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

    বাবে (int64_t grp = thread_id; grp < total_groups; grp += অঙ্ক_threads) {
        ধ্রুৱ int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        যদি (row_end > total_rows) {
            row_end = total_rows;
        }

        বাবে (int64_t row = row_start; row < row_end; ++row) {
            int64_t i1 = row % ne1;
            int64_t i2 = (row / ne1) % ne2;
            int64_t i3 = row / (ne1 * ne2);

            ধ্রুৱ ভিন্ন * src_row = (ধ্রুৱ ভিন্ন *) ((ধ্রুৱ ৱর্ণ *) src0_data + i1 * snb1 + i2 * snb2 + i3 * snb3);
            ভিন্ন *       dst_row = (ভিন্ন *) ((ৱর্ণ *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            ভিন্ন acc = 0.0f;
            বাবে (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_row[i0];
                dst_row[i0] = acc;
            }
        }
    }

    ঘূৰাই 0;
}
