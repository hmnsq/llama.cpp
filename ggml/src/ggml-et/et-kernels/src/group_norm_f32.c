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

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ಸಂರಚನಾ ggml_et_group_norm_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor dst;
    int32_t            n_groups;
    ಭಿನ್ನ              eps;
};

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_group_norm_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಯದಿ (!kernel_env) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪೂರ್ಣಾಙ್ಕ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_threads = get_ಅಂಕೆ_threads(kernel_env->shire_mask);

    ಯದಿ (thread_id < 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಯದಿ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಸಂರಚನಾ ggml_tensor * src0 = &params->src0;
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;

    ಯದಿ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ ಭಿನ್ನ * src0_data = (ಧ್ರುವ ಭಿನ್ನ *) src0->data;
    ಭಿನ್ನ *       dst_data  = (ಭಿನ್ನ *) dst->data;

    ಯದಿ (!src0_data || !dst_data) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ int32_t n_groups = params->n_groups;
    ಧ್ರುವ ಭಿನ್ನ   eps      = params->eps;

    ಯದಿ (n_groups <= 0 || eps < 0.0f) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ int64_t ne0 = dst->ne[0];
    ಧ್ರುವ int64_t ne1 = dst->ne[1];
    ಧ್ರುವ int64_t ne2 = dst->ne[2];
    ಧ್ರುವ int64_t ne3 = dst->ne[3];

    ಯದಿ (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ int64_t nb1  = dst->nb[1];
    ಧ್ರುವ int64_t nb2  = dst->nb[2];
    ಧ್ರುವ int64_t nb3  = dst->nb[3];
    ಧ್ರುವ int64_t nb01 = src0->nb[1];
    ಧ್ರುವ int64_t nb02 = src0->nb[2];
    ಧ್ರುವ int64_t nb03 = src0->nb[3];

    ಧ್ರುವ int64_t channels_per_group = (ne2 + n_groups - 1) / n_groups;
    ಯದಿ (channels_per_group <= 0) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ int64_t active_groups        = (ne2 + channels_per_group - 1) / channels_per_group;
    ಧ್ರುವ int64_t total_work           = active_groups * ne3;
    ಧ್ರುವ int64_t rows_per_write_group = et_rows_per_cacheline_group(ne0, ಪರಿಮಾಣ(ಭಿನ್ನ));

    ಪ್ರತಿ (int64_t work = thread_id; work < total_work; work += ಅಂಕೆ_threads) {
        ಧ್ರುವ int64_t i3        = work / active_groups;
        ಧ್ರುವ int64_t group_idx = work % active_groups;

        ಧ್ರುವ int64_t channel_start = group_idx * channels_per_group;
        int64_t       channel_end   = channel_start + channels_per_group;
        ಯದಿ (channel_end > ne2) {
            channel_end = ne2;
        }

        ಧ್ರುವ int64_t channel_ಎಣಿಕೆ = channel_end - channel_start;
        ಯದಿ (channel_ಎಣಿಕೆ <= 0) {
            ಅನುವರ್ತ;
        }

        ಭಿನ್ನ sum   = 0.0f;
        ಭಿನ್ನ denom = 0.0f;
        ಪ್ರತಿ (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            ಪ್ರತಿ (int64_t i1 = 0; i1 < ne1; ++i1) {
                ಧ್ರುವ ಭಿನ್ನ * src_row = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                ಪ್ರತಿ (int64_t i0 = 0; i0 < ne0; ++i0) {
                    sum += src_row[i0];
                    denom += 1.0f;
                }
            }
        }

        ಧ್ರುವ ಭಿನ್ನ mean = et_fdiv(sum, denom);

        ಭಿನ್ನ var_sum = 0.0f;
        ಪ್ರತಿ (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            ಪ್ರತಿ (int64_t i1 = 0; i1 < ne1; ++i1) {
                ಧ್ರುವ ಭಿನ್ನ * src_row = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                ಪ್ರತಿ (int64_t i0 = 0; i0 < ne0; ++i0) {
                    ಧ್ರುವ ಭಿನ್ನ centered = src_row[i0] - mean;
                    var_sum += centered * centered;
                }
            }
        }

        ಧ್ರುವ ಭಿನ್ನ variance = et_fdiv(var_sum, denom);
        ಧ್ರುವ ಭಿನ್ನ scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        ಯದಿ (ne0 % 16 == 0) {
            ಪ್ರತಿ (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
                ಪ್ರತಿ (int64_t i1 = 0; i1 < ne1; ++i1) {
                    ಧ್ರುವ ಭಿನ್ನ * src_row =
                        (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    ಭಿನ್ನ * dst_row = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    ಪ್ರತಿ (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        } ಅನ್ಯಥಾ {
            ಧ್ರುವ int64_t total_rows_in_group = channel_ಎಣಿಕೆ * ne1;
            ಧ್ರುವ int64_t total_write_groups  = (total_rows_in_group + rows_per_write_group - 1) / rows_per_write_group;

            ಪ್ರತಿ (int64_t write_group = 0; write_group < total_write_groups; ++write_group) {
                ಧ್ರುವ int64_t row_start = write_group * rows_per_write_group;
                int64_t       row_end   = row_start + rows_per_write_group;
                ಯದಿ (row_end > total_rows_in_group) {
                    row_end = total_rows_in_group;
                }

                ಪ್ರತಿ (int64_t row = row_start; row < row_end; ++row) {
                    ಧ್ರುವ int64_t local_i2 = row / ne1;
                    ಧ್ರುವ int64_t i1       = row % ne1;
                    ಧ್ರುವ int64_t i2       = channel_start + local_i2;

                    ಧ್ರುವ ಭಿನ್ನ * src_row =
                        (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    ಭಿನ್ನ * dst_row = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    ಪ್ರತಿ (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        }
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}
