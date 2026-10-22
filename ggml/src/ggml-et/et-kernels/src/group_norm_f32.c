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

#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "math_fp.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <stdint.h>

ഘടന ggml_et_group_norm_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor dst;
    int32_t            n_groups;
    ദശാംശം              eps;
};

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_group_norm_params * params, ശൂന്യം * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    എങ്കിൽ (!kernel_env) {
        തിരികെ -1;
    }

    പൂർണ്ണസംഖ്യ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    പൂർണ്ണസംഖ്യ അക്കം_threads = get_അക്കം_threads(kernel_env->shire_mask);

    എങ്കിൽ (thread_id < 0) {
        തിരികെ 0;
    }

    എങ്കിൽ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        തിരികെ -1;
    }

    ഘടന ggml_tensor * src0 = &params->src0;
    ഘടന ggml_tensor * dst  = &params->dst;

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        തിരികെ -1;
    }

    സ്ഥിരം ദശാംശം * src0_data = (സ്ഥിരം ദശാംശം *) src0->data;
    ദശാംശം *       dst_data  = (ദശാംശം *) dst->data;

    എങ്കിൽ (!src0_data || !dst_data) {
        തിരികെ -1;
    }

    സ്ഥിരം int32_t n_groups = params->n_groups;
    സ്ഥിരം ദശാംശം   eps      = params->eps;

    എങ്കിൽ (n_groups <= 0 || eps < 0.0f) {
        തിരികെ -1;
    }

    സ്ഥിരം int64_t ne0 = dst->ne[0];
    സ്ഥിരം int64_t ne1 = dst->ne[1];
    സ്ഥിരം int64_t ne2 = dst->ne[2];
    സ്ഥിരം int64_t ne3 = dst->ne[3];

    എങ്കിൽ (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        തിരികെ -1;
    }

    സ്ഥിരം int64_t nb1  = dst->nb[1];
    സ്ഥിരം int64_t nb2  = dst->nb[2];
    സ്ഥിരം int64_t nb3  = dst->nb[3];
    സ്ഥിരം int64_t nb01 = src0->nb[1];
    സ്ഥിരം int64_t nb02 = src0->nb[2];
    സ്ഥിരം int64_t nb03 = src0->nb[3];

    സ്ഥിരം int64_t channels_per_group = (ne2 + n_groups - 1) / n_groups;
    എങ്കിൽ (channels_per_group <= 0) {
        തിരികെ -1;
    }

    സ്ഥിരം int64_t active_groups        = (ne2 + channels_per_group - 1) / channels_per_group;
    സ്ഥിരം int64_t total_work           = active_groups * ne3;
    സ്ഥിരം int64_t rows_per_write_group = et_rows_per_cacheline_group(ne0, വലുപ്പം(ദശാംശം));

    വേണ്ടി (int64_t work = thread_id; work < total_work; work += അക്കം_threads) {
        സ്ഥിരം int64_t i3        = work / active_groups;
        സ്ഥിരം int64_t group_idx = work % active_groups;

        സ്ഥിരം int64_t channel_start = group_idx * channels_per_group;
        int64_t       channel_end   = channel_start + channels_per_group;
        എങ്കിൽ (channel_end > ne2) {
            channel_end = ne2;
        }

        സ്ഥിരം int64_t channel_എണ്ണം = channel_end - channel_start;
        എങ്കിൽ (channel_എണ്ണം <= 0) {
            തുടരുക;
        }

        ദശാംശം sum   = 0.0f;
        ദശാംശം denom = 0.0f;
        വേണ്ടി (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            വേണ്ടി (int64_t i1 = 0; i1 < ne1; ++i1) {
                സ്ഥിരം ദശാംശം * src_row = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                വേണ്ടി (int64_t i0 = 0; i0 < ne0; ++i0) {
                    sum += src_row[i0];
                    denom += 1.0f;
                }
            }
        }

        സ്ഥിരം ദശാംശം mean = et_fdiv(sum, denom);

        ദശാംശം var_sum = 0.0f;
        വേണ്ടി (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            വേണ്ടി (int64_t i1 = 0; i1 < ne1; ++i1) {
                സ്ഥിരം ദശാംശം * src_row = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                വേണ്ടി (int64_t i0 = 0; i0 < ne0; ++i0) {
                    സ്ഥിരം ദശാംശം centered = src_row[i0] - mean;
                    var_sum += centered * centered;
                }
            }
        }

        സ്ഥിരം ദശാംശം variance = et_fdiv(var_sum, denom);
        സ്ഥിരം ദശാംശം scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        എങ്കിൽ (ne0 % 16 == 0) {
            വേണ്ടി (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
                വേണ്ടി (int64_t i1 = 0; i1 < ne1; ++i1) {
                    സ്ഥിരം ദശാംശം * src_row =
                        (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    ദശാംശം * dst_row = (ദശാംശം *) ((അക്ഷരം *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    വേണ്ടി (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        } അല്ലെങ്കിൽ {
            സ്ഥിരം int64_t total_rows_in_group = channel_എണ്ണം * ne1;
            സ്ഥിരം int64_t total_write_groups  = (total_rows_in_group + rows_per_write_group - 1) / rows_per_write_group;

            വേണ്ടി (int64_t write_group = 0; write_group < total_write_groups; ++write_group) {
                സ്ഥിരം int64_t row_start = write_group * rows_per_write_group;
                int64_t       row_end   = row_start + rows_per_write_group;
                എങ്കിൽ (row_end > total_rows_in_group) {
                    row_end = total_rows_in_group;
                }

                വേണ്ടി (int64_t row = row_start; row < row_end; ++row) {
                    സ്ഥിരം int64_t local_i2 = row / ne1;
                    സ്ഥിരം int64_t i1       = row % ne1;
                    സ്ഥിരം int64_t i2       = channel_start + local_i2;

                    സ്ഥിരം ദശാംശം * src_row =
                        (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    ദശാംശം * dst_row = (ദശാംശം *) ((അക്ഷരം *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    വേണ്ടി (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        }
    }

    തിരികെ 0;
}
