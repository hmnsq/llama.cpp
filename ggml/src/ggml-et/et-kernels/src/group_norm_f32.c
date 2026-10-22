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
    दशांश              eps;
};

पूर्णांक entry_point(संरचना ggml_et_group_norm_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जर (!kernel_env) {
        परत -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    जर (thread_id < 0) {
        परत 0;
    }

    जर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        परत -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    जर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        परत -1;
    }

    स्थिरांक दशांश * src0_data = (स्थिरांक दशांश *) src0->data;
    दशांश *       dst_data  = (दशांश *) dst->data;

    जर (!src0_data || !dst_data) {
        परत -1;
    }

    स्थिरांक int32_t n_groups = params->n_groups;
    स्थिरांक दशांश   eps      = params->eps;

    जर (n_groups <= 0 || eps < 0.0f) {
        परत -1;
    }

    स्थिरांक int64_t ne0 = dst->ne[0];
    स्थिरांक int64_t ne1 = dst->ne[1];
    स्थिरांक int64_t ne2 = dst->ne[2];
    स्थिरांक int64_t ne3 = dst->ne[3];

    जर (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        परत -1;
    }

    स्थिरांक int64_t nb1  = dst->nb[1];
    स्थिरांक int64_t nb2  = dst->nb[2];
    स्थिरांक int64_t nb3  = dst->nb[3];
    स्थिरांक int64_t nb01 = src0->nb[1];
    स्थिरांक int64_t nb02 = src0->nb[2];
    स्थिरांक int64_t nb03 = src0->nb[3];

    स्थिरांक int64_t channels_per_group = (ne2 + n_groups - 1) / n_groups;
    जर (channels_per_group <= 0) {
        परत -1;
    }

    स्थिरांक int64_t active_groups        = (ne2 + channels_per_group - 1) / channels_per_group;
    स्थिरांक int64_t total_work           = active_groups * ne3;
    स्थिरांक int64_t rows_per_write_group = et_rows_per_cacheline_group(ne0, आकार(दशांश));

    साठी (int64_t work = thread_id; work < total_work; work += अंक_threads) {
        स्थिरांक int64_t i3        = work / active_groups;
        स्थिरांक int64_t group_idx = work % active_groups;

        स्थिरांक int64_t channel_start = group_idx * channels_per_group;
        int64_t       channel_end   = channel_start + channels_per_group;
        जर (channel_end > ne2) {
            channel_end = ne2;
        }

        स्थिरांक int64_t channel_मोजणी = channel_end - channel_start;
        जर (channel_मोजणी <= 0) {
            सुरूठेवा;
        }

        दशांश sum   = 0.0f;
        दशांश denom = 0.0f;
        साठी (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            साठी (int64_t i1 = 0; i1 < ne1; ++i1) {
                स्थिरांक दशांश * src_row = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                साठी (int64_t i0 = 0; i0 < ne0; ++i0) {
                    sum += src_row[i0];
                    denom += 1.0f;
                }
            }
        }

        स्थिरांक दशांश mean = et_fdiv(sum, denom);

        दशांश var_sum = 0.0f;
        साठी (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
            साठी (int64_t i1 = 0; i1 < ne1; ++i1) {
                स्थिरांक दशांश * src_row = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                साठी (int64_t i0 = 0; i0 < ne0; ++i0) {
                    स्थिरांक दशांश centered = src_row[i0] - mean;
                    var_sum += centered * centered;
                }
            }
        }

        स्थिरांक दशांश variance = et_fdiv(var_sum, denom);
        स्थिरांक दशांश scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        जर (ne0 % 16 == 0) {
            साठी (int64_t i2 = channel_start; i2 < channel_end; ++i2) {
                साठी (int64_t i1 = 0; i1 < ne1; ++i1) {
                    स्थिरांक दशांश * src_row =
                        (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    दशांश * dst_row = (दशांश *) ((अक्षर *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    साठी (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        } अन्यथा {
            स्थिरांक int64_t total_rows_in_group = channel_मोजणी * ne1;
            स्थिरांक int64_t total_write_groups  = (total_rows_in_group + rows_per_write_group - 1) / rows_per_write_group;

            साठी (int64_t write_group = 0; write_group < total_write_groups; ++write_group) {
                स्थिरांक int64_t row_start = write_group * rows_per_write_group;
                int64_t       row_end   = row_start + rows_per_write_group;
                जर (row_end > total_rows_in_group) {
                    row_end = total_rows_in_group;
                }

                साठी (int64_t row = row_start; row < row_end; ++row) {
                    स्थिरांक int64_t local_i2 = row / ne1;
                    स्थिरांक int64_t i1       = row % ne1;
                    स्थिरांक int64_t i2       = channel_start + local_i2;

                    स्थिरांक दशांश * src_row =
                        (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src0_data + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    दशांश * dst_row = (दशांश *) ((अक्षर *) dst_data + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    साठी (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - mean) * scale;
                    }
                }
            }
        }
    }

    परत 0;
}
