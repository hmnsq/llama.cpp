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

    यदि (!kernel_env) {
        फर्काऊ -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अङ्क_threads = get_अङ्क_threads(kernel_env->shire_mask);

    यदि (thread_id < 0) {
        फर्काऊ 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        फर्काऊ -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        फर्काऊ -1;
    }

    अचर भिन्न * src0_data = (अचर भिन्न *) src0->data;
    भिन्न *       dst_data  = (भिन्न *) dst->data;

    यदि (!src0_data || !dst_data) {
        फर्काऊ -1;
    }

    अचर int32_t n_groups = params->n_groups;
    अचर भिन्न   eps      = params->eps;

    यदि (n_groups <= 0 || eps < 0.0f) {
        फर्काऊ -1;
    }

    अचर int64_t ne0 = dst->ne[0];
    अचर int64_t ne1 = dst->ne[1];
    अचर int64_t ne2 = dst->ne[2];
    अचर int64_t ne3 = dst->ne[3];

    यदि (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        फर्काऊ -1;
    }

    अचर int64_t nb1  = dst->nb[1];
    अचर int64_t nb2  = dst->nb[2];
    अचर int64_t nb3  = dst->nb[3];
    अचर int64_t nb01 = src0->nb[1];
    अचर int64_t nb02 = src0->nb[2];
    अचर int64_t nb03 = src0->nb[3];

    अचर int64_t channels_per_group = (ne2 + n_groups - 1) / n_groups;
    यदि (channels_per_group <= 0) {
        फर्काऊ -1;
    }

    अचर int64_t active_groups        = (ne2 + channels_per_group - 1) / channels_per_group;
    अचर int64_t total_work           = active_groups * ne3;
    अचर int64_t rows_per_write_group = et_rows_per_cacheline_group(ne0, परिमाण(भिन्न));

    लागि (int64_t work = thread_id; work < total_work; work += अङ्क_threads) {
        अचर int64_t i3        = work / active_groups;
        अचर int64_t group_idx = work % active_groups;

        अचर int64_t channel_start = group_idx * channels_per_group;
        int64_t       channel_end   = channel_start + channels_per_group;
        यदि (channel_end > ne2) {
            channel_end = ne2;
        }

        अचर int64_t channel_गन्ती = channel_end - channel_start;
        यदि (channel_गन्ती <= 0) {
            जारी;
        }

        भिन्न sum   = 0.0f;
        भिन्न denom = 0.0f;
        लागि (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            लागि (int64_t i1 = 0; i1 < ne1; ++i1) {
                अचर भिन्न * src_row = (अचर भिन्न *) ((अचर वर्ण *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                लागि (int64_t i0 = 0; i0 < ne0; ++i0) {
                    sum += src_row[i0];
                    denom += 1.0f;
                }
            }
        }

        अचर भिन्न mean = et_fdiv(sum, denom);

        भिन्न var_sum = 0.0f;
        लागि (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            लागि (int64_t i1 = 0; i1 < ne1; ++i1) {
                अचर भिन्न * src_row = (अचर भिन्न *) ((अचर वर्ण *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                लागि (int64_t i0 = 0; i0 < ne0; ++i0) {
                    अचर भिन्न centered = src_row[i0] - mean;
                    var_sum += centered * centered;
                }
            }
        }

        अचर भिन्न variance = et_fdiv(var_sum, denom);
        अचर भिन्न scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        यदि (ne0 % 16 == 0) {
            लागि (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
                लागि (int64_t i1 = 0; i1 < ne1; ++i1) {
                    अचर भिन्न * src_row =
                        (अचर भिन्न *) ((अचर वर्ण *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    भिन्न * dst_row = (भिन्न *) ((वर्ण *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    लागि (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        } अन्यथा {
            अचर int64_t total_rows_in_group = channel_गन्ती * ne1;
            अचर int64_t total_write_groups  = (total_rows_in_group + rows_per_write_group - 1) / rows_per_write_group;

            लागि (int64_t write_group = 0; write_group < total_write_groups; ++write_group) {
                अचर int64_t row_start = write_group * rows_per_write_group;
                int64_t       row_end   = row_start + rows_per_write_group;
                यदि (row_end > total_rows_in_group) {
                    row_end = total_rows_in_group;
                }

                लागि (int64_t row = row_start; row < row_end; ++row) {
                    अचर int64_t local_i2 = row / ne1;
                    अचर int64_t i1       = row % ne1;
                    अचर int64_t i2       = channel_start + local_i2;

                    अचर भिन्न * src_row =
                        (अचर भिन्न *) ((अचर वर्ण *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    भिन्न * dst_row = (भिन्न *) ((वर्ण *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    लागि (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        }
    }

    फर्काऊ 0;
}
