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

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "math_fp.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>

ସଂରଚନା ggml_et_group_norm_params {
    ସଂରଚନା ggml_tensor src0;
    ସଂରଚନା ggml_tensor dst;
    int32_t            n_groups;
    ଦଶମିକ              eps;
};

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_group_norm_params * params, ଶୂନ୍ୟ * env) {
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

    ଅଚଳ ଦଶମିକ * src0_data = (ଅଚଳ ଦଶମିକ *) src0->data;
    ଦଶମିକ *       dst_data  = (ଦଶମିକ *) dst->data;

    ଯଦି (!src0_data || !dst_data) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int32_t n_groups = params->n_groups;
    ଅଚଳ ଦଶମିକ   eps      = params->eps;

    ଯଦି (n_groups <= 0 || eps < 0.0f) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int64_t ne0 = dst->ne[0];
    ଅଚଳ int64_t ne1 = dst->ne[1];
    ଅଚଳ int64_t ne2 = dst->ne[2];
    ଅଚଳ int64_t ne3 = dst->ne[3];

    ଯଦି (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int64_t nb1  = dst->nb[1];
    ଅଚଳ int64_t nb2  = dst->nb[2];
    ଅଚଳ int64_t nb3  = dst->nb[3];
    ଅଚଳ int64_t nb01 = src0->nb[1];
    ଅଚଳ int64_t nb02 = src0->nb[2];
    ଅଚଳ int64_t nb03 = src0->nb[3];

    ଅଚଳ int64_t channels_per_group = (ne2 + n_groups - 1) / n_groups;
    ଯଦି (channels_per_group <= 0) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int64_t active_groups        = (ne2 + channels_per_group - 1) / channels_per_group;
    ଅଚଳ int64_t total_work           = active_groups * ne3;
    ଅଚଳ int64_t rows_per_write_group = et_rows_per_cacheline_group(ne0, ଆକାର(ଦଶମିକ));

    ପାଇଁ (int64_t work = thread_id; work < total_work; work += ଅଙ୍କ_threads) {
        ଅଚଳ int64_t i3        = work / active_groups;
        ଅଚଳ int64_t group_idx = work % active_groups;

        ଅଚଳ int64_t channel_start = group_idx * channels_per_group;
        int64_t       channel_end   = channel_start + channels_per_group;
        ଯଦି (channel_end > ne2) {
            channel_end = ne2;
        }

        ଅଚଳ int64_t channel_ଗଣନା = channel_end - channel_start;
        ଯଦି (channel_ଗଣନା <= 0) {
            ଜାରି;
        }

        ଦଶମିକ sum   = 0.0f;
        ଦଶମିକ denom = 0.0f;
        ପାଇଁ (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            ପାଇଁ (int64_t i1 = 0; i1 < ne1; ++i1) {
                ଅଚଳ ଦଶମିକ * src_row = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                ପାଇଁ (int64_t i0 = 0; i0 < ne0; ++i0) {
                    sum += src_row[i0];
                    denom += 1.0f;
                }
            }
        }

        ଅଚଳ ଦଶମିକ mean = et_fdiv(sum, denom);

        ଦଶମିକ var_sum = 0.0f;
        ପାଇଁ (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            ପାଇଁ (int64_t i1 = 0; i1 < ne1; ++i1) {
                ଅଚଳ ଦଶମିକ * src_row = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                ପାଇଁ (int64_t i0 = 0; i0 < ne0; ++i0) {
                    ଅଚଳ ଦଶମିକ centered = src_row[i0] - mean;
                    var_sum += centered * centered;
                }
            }
        }

        ଅଚଳ ଦଶମିକ variance = et_fdiv(var_sum, denom);
        ଅଚଳ ଦଶମିକ scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        ଯଦି (ne0 % 16 == 0) {
            ପାଇଁ (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
                ପାଇଁ (int64_t i1 = 0; i1 < ne1; ++i1) {
                    ଅଚଳ ଦଶମିକ * src_row =
                        (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    ଦଶମିକ * dst_row = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    ପାଇଁ (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        } ଅନ୍ୟଥା {
            ଅଚଳ int64_t total_rows_in_group = channel_ଗଣନା * ne1;
            ଅଚଳ int64_t total_write_groups  = (total_rows_in_group + rows_per_write_group - 1) / rows_per_write_group;

            ପାଇଁ (int64_t write_group = 0; write_group < total_write_groups; ++write_group) {
                ଅଚଳ int64_t row_start = write_group * rows_per_write_group;
                int64_t       row_end   = row_start + rows_per_write_group;
                ଯଦି (row_end > total_rows_in_group) {
                    row_end = total_rows_in_group;
                }

                ପାଇଁ (int64_t row = row_start; row < row_end; ++row) {
                    ଅଚଳ int64_t local_i2 = row / ne1;
                    ଅଚଳ int64_t i1       = row % ne1;
                    ଅଚଳ int64_t i2       = channel_start + local_i2;

                    ଅଚଳ ଦଶମିକ * src_row =
                        (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    ଦଶମିକ * dst_row = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    ପାଇଁ (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        }
    }

    ଫେରନ୍ତୁ 0;
}
