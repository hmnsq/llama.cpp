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

#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_group_norm_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor dst;
    int32_t            n_groups;
    عائم              eps;
};

صحيح entry_point(بنية ggml_et_group_norm_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحيح عدد_threads = get_عدد_threads(kernel_env->shire_mask);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    ثابت عائم * src0_data = (ثابت عائم *) src0->data;
    عائم *       dst_data  = (عائم *) dst->data;

    إذا (!src0_data || !dst_data) {
        أرجع -1;
    }

    ثابت int32_t n_groups = params->n_groups;
    ثابت عائم   eps      = params->eps;

    إذا (n_groups <= 0 || eps < 0.0f) {
        أرجع -1;
    }

    ثابت int64_t ne0 = dst->ne[0];
    ثابت int64_t ne1 = dst->ne[1];
    ثابت int64_t ne2 = dst->ne[2];
    ثابت int64_t ne3 = dst->ne[3];

    إذا (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        أرجع -1;
    }

    ثابت int64_t nb1  = dst->nb[1];
    ثابت int64_t nb2  = dst->nb[2];
    ثابت int64_t nb3  = dst->nb[3];
    ثابت int64_t nb01 = src0->nb[1];
    ثابت int64_t nb02 = src0->nb[2];
    ثابت int64_t nb03 = src0->nb[3];

    ثابت int64_t channels_per_group = (ne2 + n_groups - 1) / n_groups;
    إذا (channels_per_group <= 0) {
        أرجع -1;
    }

    ثابت int64_t active_groups        = (ne2 + channels_per_group - 1) / channels_per_group;
    ثابت int64_t total_work           = active_groups * ne3;
    ثابت int64_t rows_per_write_group = et_rows_per_cacheline_group(ne0, حجم(عائم));

    لكل (int64_t work = thread_id; work < total_work; work += عدد_threads) {
        ثابت int64_t i3        = work / active_groups;
        ثابت int64_t group_idx = work % active_groups;

        ثابت int64_t channel_start = group_idx * channels_per_group;
        int64_t       channel_end   = channel_start + channels_per_group;
        إذا (channel_end > ne2) {
            channel_end = ne2;
        }

        ثابت int64_t channel_گنتی = channel_end - channel_start;
        إذا (channel_گنتی <= 0) {
            استمر;
        }

        عائم sum   = 0.0f;
        عائم denom = 0.0f;
        لكل (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            لكل (int64_t i1 = 0; i1 < ne1; ++i1) {
                ثابت عائم * src_row = (ثابت عائم *) ((ثابت حرف *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                    sum += src_row[i0];
                    denom += 1.0f;
                }
            }
        }

        ثابت عائم mean = et_fdiv(sum, denom);

        عائم var_sum = 0.0f;
        لكل (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            لكل (int64_t i1 = 0; i1 < ne1; ++i1) {
                ثابت عائم * src_row = (ثابت عائم *) ((ثابت حرف *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                    ثابت عائم centered = src_row[i0] - mean;
                    var_sum += centered * centered;
                }
            }
        }

        ثابت عائم variance = et_fdiv(var_sum, denom);
        ثابت عائم scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        إذا (ne0 % 16 == 0) {
            لكل (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
                لكل (int64_t i1 = 0; i1 < ne1; ++i1) {
                    ثابت عائم * src_row =
                        (ثابت عائم *) ((ثابت حرف *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    عائم * dst_row = (عائم *) ((حرف *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        } وإلا {
            ثابت int64_t total_rows_in_group = channel_گنتی * ne1;
            ثابت int64_t total_write_groups  = (total_rows_in_group + rows_per_write_group - 1) / rows_per_write_group;

            لكل (int64_t write_group = 0; write_group < total_write_groups; ++write_group) {
                ثابت int64_t row_start = write_group * rows_per_write_group;
                int64_t       row_end   = row_start + rows_per_write_group;
                إذا (row_end > total_rows_in_group) {
                    row_end = total_rows_in_group;
                }

                لكل (int64_t row = row_start; row < row_end; ++row) {
                    ثابت int64_t local_i2 = row / ne1;
                    ثابت int64_t i1       = row % ne1;
                    ثابت int64_t i2       = channel_start + local_i2;

                    ثابت عائم * src_row =
                        (ثابت عائم *) ((ثابت حرف *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    عائم * dst_row = (عائم *) ((حرف *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        }
    }

    أرجع 0;
}
