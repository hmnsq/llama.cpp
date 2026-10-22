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

#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <stdint.h>

ساختار ggml_et_group_norm_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor dst;
    int32_t            n_groups;
    اعشاری              eps;
};

صحیح entry_point(ساختار ggml_et_group_norm_params * params, تهی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        بازگشت -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحیح رقم_threads = get_رقم_threads(kernel_env->shire_mask);

    اگر (thread_id < 0) {
        بازگشت 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        بازگشت -1;
    }

    ساختار ggml_tensor * src0 = &params->src0;
    ساختار ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        بازگشت -1;
    }

    ثابت اعشاری * src0_data = (ثابت اعشاری *) src0->data;
    اعشاری *       dst_data  = (اعشاری *) dst->data;

    اگر (!src0_data || !dst_data) {
        بازگشت -1;
    }

    ثابت int32_t n_groups = params->n_groups;
    ثابت اعشاری   eps      = params->eps;

    اگر (n_groups <= 0 || eps < 0.0f) {
        بازگشت -1;
    }

    ثابت int64_t ne0 = dst->ne[0];
    ثابت int64_t ne1 = dst->ne[1];
    ثابت int64_t ne2 = dst->ne[2];
    ثابت int64_t ne3 = dst->ne[3];

    اگر (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        بازگشت -1;
    }

    ثابت int64_t nb1  = dst->nb[1];
    ثابت int64_t nb2  = dst->nb[2];
    ثابت int64_t nb3  = dst->nb[3];
    ثابت int64_t nb01 = src0->nb[1];
    ثابت int64_t nb02 = src0->nb[2];
    ثابت int64_t nb03 = src0->nb[3];

    ثابت int64_t channels_per_group = (ne2 + n_groups - 1) / n_groups;
    اگر (channels_per_group <= 0) {
        بازگشت -1;
    }

    ثابت int64_t active_groups        = (ne2 + channels_per_group - 1) / channels_per_group;
    ثابت int64_t total_work           = active_groups * ne3;
    ثابت int64_t rows_per_write_group = et_rows_per_cacheline_group(ne0, اندازه(اعشاری));

    برای (int64_t work = thread_id; work < total_work; work += رقم_threads) {
        ثابت int64_t i3        = work / active_groups;
        ثابت int64_t group_idx = work % active_groups;

        ثابت int64_t channel_start = group_idx * channels_per_group;
        int64_t       channel_end   = channel_start + channels_per_group;
        اگر (channel_end > ne2) {
            channel_end = ne2;
        }

        ثابت int64_t channel_شمار = channel_end - channel_start;
        اگر (channel_شمار <= 0) {
            ادامه;
        }

        اعشاری sum   = 0.0f;
        اعشاری denom = 0.0f;
        برای (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            برای (int64_t i1 = 0; i1 < ne1; ++i1) {
                ثابت اعشاری * src_row = (ثابت اعشاری *) ((ثابت نویسه *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                برای (int64_t i0 = 0; i0 < ne0; ++i0) {
                    sum += src_row[i0];
                    denom += 1.0f;
                }
            }
        }

        ثابت اعشاری mean = et_fdiv(sum, denom);

        اعشاری var_sum = 0.0f;
        برای (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            برای (int64_t i1 = 0; i1 < ne1; ++i1) {
                ثابت اعشاری * src_row = (ثابت اعشاری *) ((ثابت نویسه *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                برای (int64_t i0 = 0; i0 < ne0; ++i0) {
                    ثابت اعشاری centered = src_row[i0] - mean;
                    var_sum += centered * centered;
                }
            }
        }

        ثابت اعشاری variance = et_fdiv(var_sum, denom);
        ثابت اعشاری scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        اگر (ne0 % 16 == 0) {
            برای (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
                برای (int64_t i1 = 0; i1 < ne1; ++i1) {
                    ثابت اعشاری * src_row =
                        (ثابت اعشاری *) ((ثابت نویسه *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    اعشاری * dst_row = (اعشاری *) ((نویسه *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    برای (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        } وگرنه {
            ثابت int64_t total_rows_in_group = channel_شمار * ne1;
            ثابت int64_t total_write_groups  = (total_rows_in_group + rows_per_write_group - 1) / rows_per_write_group;

            برای (int64_t write_group = 0; write_group < total_write_groups; ++write_group) {
                ثابت int64_t row_start = write_group * rows_per_write_group;
                int64_t       row_end   = row_start + rows_per_write_group;
                اگر (row_end > total_rows_in_group) {
                    row_end = total_rows_in_group;
                }

                برای (int64_t row = row_start; row < row_end; ++row) {
                    ثابت int64_t local_i2 = row / ne1;
                    ثابت int64_t i1       = row % ne1;
                    ثابت int64_t i2       = channel_start + local_i2;

                    ثابت اعشاری * src_row =
                        (ثابت اعشاری *) ((ثابت نویسه *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    اعشاری * dst_row = (اعشاری *) ((نویسه *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    برای (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        }
    }

    بازگشت 0;
}
