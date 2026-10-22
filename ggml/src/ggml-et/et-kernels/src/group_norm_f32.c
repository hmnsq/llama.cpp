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

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "math_fp.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <stdint.h>

સંરચના ggml_et_group_norm_params {
    સંરચના ggml_tensor src0;
    સંરચના ggml_tensor dst;
    int32_t            n_groups;
    દશાંશ              eps;
};

પૂર્ણાંક entry_point(સંરચના ggml_et_group_norm_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    જો (!kernel_env) {
        પાછા -1;
    }

    પૂર્ણાંક thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    પૂર્ણાંક અંક_threads = get_અંક_threads(kernel_env->shire_mask);

    જો (thread_id < 0) {
        પાછા 0;
    }

    જો (params == 0 || ((uint64_t) params & 0x7) != 0) {
        પાછા -1;
    }

    સંરચના ggml_tensor * src0 = &params->src0;
    સંરચના ggml_tensor * dst  = &params->dst;

    જો (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા -1;
    }

    અચલ દશાંશ * src0_data = (અચલ દશાંશ *) src0->data;
    દશાંશ *       dst_data  = (દશાંશ *) dst->data;

    જો (!src0_data || !dst_data) {
        પાછા -1;
    }

    અચલ int32_t n_groups = params->n_groups;
    અચલ દશાંશ   eps      = params->eps;

    જો (n_groups <= 0 || eps < 0.0f) {
        પાછા -1;
    }

    અચલ int64_t ne0 = dst->ne[0];
    અચલ int64_t ne1 = dst->ne[1];
    અચલ int64_t ne2 = dst->ne[2];
    અચલ int64_t ne3 = dst->ne[3];

    જો (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        પાછા -1;
    }

    અચલ int64_t nb1  = dst->nb[1];
    અચલ int64_t nb2  = dst->nb[2];
    અચલ int64_t nb3  = dst->nb[3];
    અચલ int64_t nb01 = src0->nb[1];
    અચલ int64_t nb02 = src0->nb[2];
    અચલ int64_t nb03 = src0->nb[3];

    અચલ int64_t channels_per_group = (ne2 + n_groups - 1) / n_groups;
    જો (channels_per_group <= 0) {
        પાછા -1;
    }

    અચલ int64_t active_groups        = (ne2 + channels_per_group - 1) / channels_per_group;
    અચલ int64_t total_work           = active_groups * ne3;
    અચલ int64_t rows_per_write_group = et_rows_per_cacheline_group(ne0, કદ(દશાંશ));

    માટે (int64_t work = thread_id; work < total_work; work += અંક_threads) {
        અચલ int64_t i3        = work / active_groups;
        અચલ int64_t group_idx = work % active_groups;

        અચલ int64_t channel_start = group_idx * channels_per_group;
        int64_t       channel_end   = channel_start + channels_per_group;
        જો (channel_end > ne2) {
            channel_end = ne2;
        }

        અચલ int64_t channel_ગણતરી = channel_end - channel_start;
        જો (channel_ગણતરી <= 0) {
            ચાલુ;
        }

        દશાંશ sum   = 0.0f;
        દશાંશ denom = 0.0f;
        માટે (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            માટે (int64_t i1 = 0; i1 < ne1; ++i1) {
                અચલ દશાંશ * src_row = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                માટે (int64_t i0 = 0; i0 < ne0; ++i0) {
                    sum += src_row[i0];
                    denom += 1.0f;
                }
            }
        }

        અચલ દશાંશ mean = et_fdiv(sum, denom);

        દશાંશ var_sum = 0.0f;
        માટે (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            માટે (int64_t i1 = 0; i1 < ne1; ++i1) {
                અચલ દશાંશ * src_row = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                માટે (int64_t i0 = 0; i0 < ne0; ++i0) {
                    અચલ દશાંશ centered = src_row[i0] - mean;
                    var_sum += centered * centered;
                }
            }
        }

        અચલ દશાંશ variance = et_fdiv(var_sum, denom);
        અચલ દશાંશ scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        જો (ne0 % 16 == 0) {
            માટે (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
                માટે (int64_t i1 = 0; i1 < ne1; ++i1) {
                    અચલ દશાંશ * src_row =
                        (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    દશાંશ * dst_row = (દશાંશ *) ((અક્ષર *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    માટે (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        } નહીંતર {
            અચલ int64_t total_rows_in_group = channel_ગણતરી * ne1;
            અચલ int64_t total_write_groups  = (total_rows_in_group + rows_per_write_group - 1) / rows_per_write_group;

            માટે (int64_t write_group = 0; write_group < total_write_groups; ++write_group) {
                અચલ int64_t row_start = write_group * rows_per_write_group;
                int64_t       row_end   = row_start + rows_per_write_group;
                જો (row_end > total_rows_in_group) {
                    row_end = total_rows_in_group;
                }

                માટે (int64_t row = row_start; row < row_end; ++row) {
                    અચલ int64_t local_i2 = row / ne1;
                    અચલ int64_t i1       = row % ne1;
                    અચલ int64_t i2       = channel_start + local_i2;

                    અચલ દશાંશ * src_row =
                        (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    દશાંશ * dst_row = (દશાંશ *) ((અક્ષર *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    માટે (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        }
    }

    પાછા 0;
}
