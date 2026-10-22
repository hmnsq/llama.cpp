//******************************************************************************
// GROUP_NORM F32 Kernel
// Baseline scalar implementation:
//   normalize over (ne0 * ne1 * channels_in_group) for each (group, batch).
//
// Parallelization:
// - Work is partitioned across (group, batch) pairs.
// - For non-cache-aligned ne0, writes are emitted in row-groups so each thread's
//   destination write footprint still spans an integer number of cache lines.
//******************************************************************************

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "math_fp.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

সংরচনা ggml_et_group_norm_params {
    সংরচনা ggml_tensor src0;
    সংরচনা ggml_tensor dst;
    int32_t            n_groups;
    ভিন্ন              eps;
};

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_group_norm_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        প্রত্যাগম -1;
    }

    পূর্ণাঙ্ক thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণাঙ্ক অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        প্রত্যাগম 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        প্রত্যাগম -1;
    }

    সংরচনা ggml_tensor * src0 = &params->src0;
    সংরচনা ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        প্রত্যাগম -1;
    }

    ধ্রুব ভিন্ন * src0_data = (ধ্রুব ভিন্ন *) src0->data;
    ভিন্ন *       dst_data  = (ভিন্ন *) dst->data;

    যদি (!src0_data || !dst_data) {
        প্রত্যাগম -1;
    }

    ধ্রুব int32_t n_groups = params->n_groups;
    ধ্রুব ভিন্ন   eps      = params->eps;

    যদি (n_groups <= 0 || eps < 0.0f) {
        প্রত্যাগম -1;
    }

    ধ্রুব int64_t ne0 = dst->ne[0];
    ধ্রুব int64_t ne1 = dst->ne[1];
    ধ্রুব int64_t ne2 = dst->ne[2];
    ধ্রুব int64_t ne3 = dst->ne[3];

    যদি (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        প্রত্যাগম -1;
    }

    ধ্রুব int64_t nb1  = dst->nb[1];
    ধ্রুব int64_t nb2  = dst->nb[2];
    ধ্রুব int64_t nb3  = dst->nb[3];
    ধ্রুব int64_t nb01 = src0->nb[1];
    ধ্রুব int64_t nb02 = src0->nb[2];
    ধ্রুব int64_t nb03 = src0->nb[3];

    ধ্রুব int64_t channels_per_group = (ne2 + n_groups - 1) / n_groups;
    যদি (channels_per_group <= 0) {
        প্রত্যাগম -1;
    }

    ধ্রুব int64_t active_groups        = (ne2 + channels_per_group - 1) / channels_per_group;
    ধ্রুব int64_t total_work           = active_groups * ne3;
    ধ্রুব int64_t rows_per_write_group = et_rows_per_cacheline_group(ne0, পরিমাণ(ভিন্ন));

    প্রতি (int64_t work = thread_id; work < total_work; work += অঙ্ক_threads) {
        ধ্রুব int64_t i3        = work / active_groups;
        ধ্রুব int64_t group_idx = work % active_groups;

        ধ্রুব int64_t channel_start = group_idx * channels_per_group;
        int64_t       channel_end   = channel_start + channels_per_group;
        যদি (channel_end > ne2) {
            channel_end = ne2;
        }

        ধ্রুব int64_t channel_গুনতি = channel_end - channel_start;
        যদি (channel_গুনতি <= 0) {
            অনুবর্ত;
        }

        ভিন্ন sum   = 0.0f;
        ভিন্ন denom = 0.0f;
        প্রতি (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            প্রতি (int64_t i1 = 0; i1 < ne1; ++i1) {
                ধ্রুব ভিন্ন * src_row = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                প্রতি (int64_t i0 = 0; i0 < ne0; ++i0) {
                    sum += src_row[i0];
                    denom += 1.0f;
                }
            }
        }

        ধ্রুব ভিন্ন mean = et_fdiv(sum, denom);

        ভিন্ন var_sum = 0.0f;
        প্রতি (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            প্রতি (int64_t i1 = 0; i1 < ne1; ++i1) {
                ধ্রুব ভিন্ন * src_row = (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                প্রতি (int64_t i0 = 0; i0 < ne0; ++i0) {
                    ধ্রুব ভিন্ন centered = src_row[i0] - mean;
                    var_sum += centered * centered;
                }
            }
        }

        ধ্রুব ভিন্ন variance = et_fdiv(var_sum, denom);
        ধ্রুব ভিন্ন scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        যদি (ne0 % 16 == 0) {
            প্রতি (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
                প্রতি (int64_t i1 = 0; i1 < ne1; ++i1) {
                    ধ্রুব ভিন্ন * src_row =
                        (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    ভিন্ন * dst_row = (ভিন্ন *) ((বর্ণ *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    প্রতি (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        } অন্যথা {
            ধ্রুব int64_t total_rows_in_group = channel_গুনতি * ne1;
            ধ্রুব int64_t total_write_groups  = (total_rows_in_group + rows_per_write_group - 1) / rows_per_write_group;

            প্রতি (int64_t write_group = 0; write_group < total_write_groups; ++write_group) {
                ধ্রুব int64_t row_start = write_group * rows_per_write_group;
                int64_t       row_end   = row_start + rows_per_write_group;
                যদি (row_end > total_rows_in_group) {
                    row_end = total_rows_in_group;
                }

                প্রতি (int64_t row = row_start; row < row_end; ++row) {
                    ধ্রুব int64_t local_i2 = row / ne1;
                    ধ্রুব int64_t i1       = row % ne1;
                    ধ্রুব int64_t i2       = channel_start + local_i2;

                    ধ্রুব ভিন্ন * src_row =
                        (ধ্রুব ভিন্ন *) ((ধ্রুব বর্ণ *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    ভিন্ন * dst_row = (ভিন্ন *) ((বর্ণ *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    প্রতি (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        }
    }

    প্রত্যাগম 0;
}
