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

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "math_fp.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <stdint.h>

கட்டமைப்பு ggml_et_group_norm_params {
    கட்டமைப்பு ggml_tensor src0;
    கட்டமைப்பு ggml_tensor dst;
    int32_t            n_groups;
    தசம              eps;
};

முழு entry_point(கட்டமைப்பு ggml_et_group_norm_params * params, வெறுமை * env) {
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

    மாறிலி தசம * src0_data = (மாறிலி தசம *) src0->data;
    தசம *       dst_data  = (தசம *) dst->data;

    எனில் (!src0_data || !dst_data) {
        திரும்பு -1;
    }

    மாறிலி int32_t n_groups = params->n_groups;
    மாறிலி தசம   eps      = params->eps;

    எனில் (n_groups <= 0 || eps < 0.0f) {
        திரும்பு -1;
    }

    மாறிலி int64_t ne0 = dst->ne[0];
    மாறிலி int64_t ne1 = dst->ne[1];
    மாறிலி int64_t ne2 = dst->ne[2];
    மாறிலி int64_t ne3 = dst->ne[3];

    எனில் (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        திரும்பு -1;
    }

    மாறிலி int64_t nb1  = dst->nb[1];
    மாறிலி int64_t nb2  = dst->nb[2];
    மாறிலி int64_t nb3  = dst->nb[3];
    மாறிலி int64_t nb01 = src0->nb[1];
    மாறிலி int64_t nb02 = src0->nb[2];
    மாறிலி int64_t nb03 = src0->nb[3];

    மாறிலி int64_t channels_per_group = (ne2 + n_groups - 1) / n_groups;
    எனில் (channels_per_group <= 0) {
        திரும்பு -1;
    }

    மாறிலி int64_t active_groups        = (ne2 + channels_per_group - 1) / channels_per_group;
    மாறிலி int64_t total_work           = active_groups * ne3;
    மாறிலி int64_t rows_per_write_group = et_rows_per_cacheline_group(ne0, அளவு(தசம));

    க்கு (int64_t work = thread_id; work < total_work; work += இலக்கம்_threads) {
        மாறிலி int64_t i3        = work / active_groups;
        மாறிலி int64_t group_idx = work % active_groups;

        மாறிலி int64_t channel_start = group_idx * channels_per_group;
        int64_t       channel_end   = channel_start + channels_per_group;
        எனில் (channel_end > ne2) {
            channel_end = ne2;
        }

        மாறிலி int64_t channel_மொத்தம் = channel_end - channel_start;
        எனில் (channel_மொத்தம் <= 0) {
            தொடர்;
        }

        தசம sum   = 0.0f;
        தசம denom = 0.0f;
        க்கு (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            க்கு (int64_t i1 = 0; i1 < ne1; ++i1) {
                மாறிலி தசம * src_row = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                க்கு (int64_t i0 = 0; i0 < ne0; ++i0) {
                    sum += src_row[i0];
                    denom += 1.0f;
                }
            }
        }

        மாறிலி தசம mean = et_fdiv(sum, denom);

        தசம var_sum = 0.0f;
        க்கு (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            க்கு (int64_t i1 = 0; i1 < ne1; ++i1) {
                மாறிலி தசம * src_row = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                க்கு (int64_t i0 = 0; i0 < ne0; ++i0) {
                    மாறிலி தசம centered = src_row[i0] - mean;
                    var_sum += centered * centered;
                }
            }
        }

        மாறிலி தசம variance = et_fdiv(var_sum, denom);
        மாறிலி தசம scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        எனில் (ne0 % 16 == 0) {
            க்கு (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
                க்கு (int64_t i1 = 0; i1 < ne1; ++i1) {
                    மாறிலி தசம * src_row =
                        (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    தசம * dst_row = (தசம *) ((எழுத்து *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    க்கு (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        } இல்லையெனில் {
            மாறிலி int64_t total_rows_in_group = channel_மொத்தம் * ne1;
            மாறிலி int64_t total_write_groups  = (total_rows_in_group + rows_per_write_group - 1) / rows_per_write_group;

            க்கு (int64_t write_group = 0; write_group < total_write_groups; ++write_group) {
                மாறிலி int64_t row_start = write_group * rows_per_write_group;
                int64_t       row_end   = row_start + rows_per_write_group;
                எனில் (row_end > total_rows_in_group) {
                    row_end = total_rows_in_group;
                }

                க்கு (int64_t row = row_start; row < row_end; ++row) {
                    மாறிலி int64_t local_i2 = row / ne1;
                    மாறிலி int64_t i1       = row % ne1;
                    மாறிலி int64_t i2       = channel_start + local_i2;

                    மாறிலி தசம * src_row =
                        (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    தசம * dst_row = (தசம *) ((எழுத்து *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    க்கு (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        }
    }

    திரும்பு 0;
}
