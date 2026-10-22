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

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "math_fp.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

संरचना ggml_et_group_norm_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor dst;
    int32_t            n_groups;
    भिन्न              eps;
};

पूर्णाङ्क entry_point(संरचना ggml_et_group_norm_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जइ (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अङ्कः_threads = get_अङ्कः_threads(kernel_env->shire_mask);

    जइ (thread_id < 0) {
        प्रत्यागम 0;
    }

    जइ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    जइ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    ध्रुव भिन्न * src0_data = (ध्रुव भिन्न *) src0->data;
    भिन्न *       dst_data  = (भिन्न *) dst->data;

    जइ (!src0_data || !dst_data) {
        प्रत्यागम -1;
    }

    ध्रुव int32_t n_groups = params->n_groups;
    ध्रुव भिन्न   eps      = params->eps;

    जइ (n_groups <= 0 || eps < 0.0f) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t ne0 = dst->ne[0];
    ध्रुव int64_t ne1 = dst->ne[1];
    ध्रुव int64_t ne2 = dst->ne[2];
    ध्रुव int64_t ne3 = dst->ne[3];

    जइ (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t nb1  = dst->nb[1];
    ध्रुव int64_t nb2  = dst->nb[2];
    ध्रुव int64_t nb3  = dst->nb[3];
    ध्रुव int64_t nb01 = src0->nb[1];
    ध्रुव int64_t nb02 = src0->nb[2];
    ध्रुव int64_t nb03 = src0->nb[3];

    ध्रुव int64_t channels_per_group = (ne2 + n_groups - 1) / n_groups;
    जइ (channels_per_group <= 0) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t active_groups        = (ne2 + channels_per_group - 1) / channels_per_group;
    ध्रुव int64_t total_work           = active_groups * ne3;
    ध्रुव int64_t rows_per_write_group = et_rows_per_cacheline_group(ne0, परिमाण(भिन्न));

    प्रति (int64_t work = thread_id; work < total_work; work += अङ्कः_threads) {
        ध्रुव int64_t i3        = work / active_groups;
        ध्रुव int64_t group_idx = work % active_groups;

        ध्रुव int64_t channel_start = group_idx * channels_per_group;
        int64_t       channel_end   = channel_start + channels_per_group;
        जइ (channel_end > ne2) {
            channel_end = ne2;
        }

        ध्रुव int64_t channel_गणनम् = channel_end - channel_start;
        जइ (channel_गणनम् <= 0) {
            अनुवर्त;
        }

        भिन्न sum   = 0.0f;
        भिन्न denom = 0.0f;
        प्रति (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            प्रति (int64_t i1 = 0; i1 < ne1; ++i1) {
                ध्रुव भिन्न * src_row = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                प्रति (int64_t i0 = 0; i0 < ne0; ++i0) {
                    sum += src_row[i0];
                    denom += 1.0f;
                }
            }
        }

        ध्रुव भिन्न mean = et_fdiv(sum, denom);

        भिन्न var_sum = 0.0f;
        प्रति (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            प्रति (int64_t i1 = 0; i1 < ne1; ++i1) {
                ध्रुव भिन्न * src_row = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                प्रति (int64_t i0 = 0; i0 < ne0; ++i0) {
                    ध्रुव भिन्न centered = src_row[i0] - mean;
                    var_sum += centered * centered;
                }
            }
        }

        ध्रुव भिन्न variance = et_fdiv(var_sum, denom);
        ध्रुव भिन्न scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        जइ (ne0 % 16 == 0) {
            प्रति (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
                प्रति (int64_t i1 = 0; i1 < ne1; ++i1) {
                    ध्रुव भिन्न * src_row =
                        (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    भिन्न * dst_row = (भिन्न *) ((वर्ण *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    प्रति (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        } अण्णहा {
            ध्रुव int64_t total_rows_in_group = channel_गणनम् * ne1;
            ध्रुव int64_t total_write_groups  = (total_rows_in_group + rows_per_write_group - 1) / rows_per_write_group;

            प्रति (int64_t write_group = 0; write_group < total_write_groups; ++write_group) {
                ध्रुव int64_t row_start = write_group * rows_per_write_group;
                int64_t       row_end   = row_start + rows_per_write_group;
                जइ (row_end > total_rows_in_group) {
                    row_end = total_rows_in_group;
                }

                प्रति (int64_t row = row_start; row < row_end; ++row) {
                    ध्रुव int64_t local_i2 = row / ne1;
                    ध्रुव int64_t i1       = row % ne1;
                    ध्रुव int64_t i2       = channel_start + local_i2;

                    ध्रुव भिन्न * src_row =
                        (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    भिन्न * dst_row = (भिन्न *) ((वर्ण *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    प्रति (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        }
    }

    प्रत्यागम 0;
}
