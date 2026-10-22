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

#समावेश "ggml_tensor.h"
#समावेश "math_fp.h"
#समावेश "platform.h"

#समावेश <stdint.h>

काष्ठा ggml_et_group_norm_params {
    काष्ठा ggml_tensor src0;
    काष्ठा ggml_tensor dst;
    int32_t            n_groups;
    भग्न              eps;
};

पूर्णांक entry_point(काष्ठा ggml_et_group_norm_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस 0;
    }

    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;
    }

    काष्ठा ggml_tensor * src0 = &params->src0;
    काष्ठा ggml_tensor * dst  = &params->dst;

    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस -1;
    }

    स्थिर भग्न * src0_तथ्य = (स्थिर भग्न *) src0->तथ्य;
    भग्न *       dst_तथ्य  = (भग्न *) dst->तथ्य;

    अगर (!src0_तथ्य || !dst_तथ्य) {
        वापस -1;
    }

    स्थिर int32_t n_groups = params->n_groups;
    स्थिर भग्न   eps      = params->eps;

    अगर (n_groups <= 0 || eps < 0.0f) {
        वापस -1;
    }

    स्थिर int64_t ne0 = dst->ne[0];
    स्थिर int64_t ne1 = dst->ne[1];
    स्थिर int64_t ne2 = dst->ne[2];
    स्थिर int64_t ne3 = dst->ne[3];

    अगर (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        वापस -1;
    }

    स्थिर int64_t nb1  = dst->nb[1];
    स्थिर int64_t nb2  = dst->nb[2];
    स्थिर int64_t nb3  = dst->nb[3];
    स्थिर int64_t nb01 = src0->nb[1];
    स्थिर int64_t nb02 = src0->nb[2];
    स्थिर int64_t nb03 = src0->nb[3];

    स्थिर int64_t channels_per_group = (ne2 + n_groups - 1) / n_groups;
    अगर (channels_per_group <= 0) {
        वापस -1;
    }

    स्थिर int64_t active_groups        = (ne2 + channels_per_group - 1) / channels_per_group;
    स्थिर int64_t योग_work           = active_groups * ne3;
    स्थिर int64_t rows_per_write_group = et_rows_per_cacheline_group(ne0, माप(भग्न));

    क्रम (int64_t work = thread_id; work < योग_work; work += अंक_threads) {
        स्थिर int64_t i3        = work / active_groups;
        स्थिर int64_t group_idx = work % active_groups;

        स्थिर int64_t channel_start = group_idx * channels_per_group;
        int64_t       channel_end   = channel_start + channels_per_group;
        अगर (channel_end > ne2) {
            channel_end = ne2;
        }

        स्थिर int64_t channel_गिनती = channel_end - channel_start;
        अगर (channel_गिनती <= 0) {
            जारी;
        }

        भग्न योग_2   = 0.0f;
        भग्न denom = 0.0f;
        क्रम (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            क्रम (int64_t i1 = 0; i1 < ne1; ++i1) {
                स्थिर भग्न * src_row = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i3 * nb03 + i2 * nb02 + i1 * nb01);
                क्रम (int64_t i0 = 0; i0 < ne0; ++i0) {
                    योग_2 += src_row[i0];
                    denom += 1.0f;
                }
            }
        }

        स्थिर भग्न mean = et_fdiv(योग_2, denom);

        भग्न var_योग = 0.0f;
        क्रम (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            क्रम (int64_t i1 = 0; i1 < ne1; ++i1) {
                स्थिर भग्न * src_row = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i3 * nb03 + i2 * nb02 + i1 * nb01);
                क्रम (int64_t i0 = 0; i0 < ne0; ++i0) {
                    स्थिर भग्न centered = src_row[i0] - mean;
                    var_योग += centered * centered;
                }
            }
        }

        स्थिर भग्न variance = et_fdiv(var_योग, denom);
        स्थिर भग्न scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        अगर (ne0 % 16 == 0) {
            क्रम (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
                क्रम (int64_t i1 = 0; i1 < ne1; ++i1) {
                    स्थिर भग्न * src_row =
                        (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    भग्न * dst_row = (भग्न *) ((अक्षर *) dst_तथ्य + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    क्रम (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        } अन्यथा {
            स्थिर int64_t योग_rows_in_group = channel_गिनती * ne1;
            स्थिर int64_t योग_write_groups  = (योग_rows_in_group + rows_per_write_group - 1) / rows_per_write_group;

            क्रम (int64_t write_group = 0; write_group < योग_write_groups; ++write_group) {
                स्थिर int64_t row_start = write_group * rows_per_write_group;
                int64_t       row_end   = row_start + rows_per_write_group;
                अगर (row_end > योग_rows_in_group) {
                    row_end = योग_rows_in_group;
                }

                क्रम (int64_t row = row_start; row < row_end; ++row) {
                    स्थिर int64_t local_i2 = row / ne1;
                    स्थिर int64_t i1       = row % ne1;
                    स्थिर int64_t i2       = channel_start + local_i2;

                    स्थिर भग्न * src_row =
                        (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    भग्न * dst_row = (भग्न *) ((अक्षर *) dst_तथ्य + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    क्रम (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        }
    }

    वापस 0;
}
