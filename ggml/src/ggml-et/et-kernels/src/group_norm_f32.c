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

संरचना ggml_et_गणः_norm_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor dst;
    int32_t            n_groups;
    भिन्न              eps;
};

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_गणः_norm_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_आवरणम्);
    पूर्णाङ्क अङ्कः_तन्तवः = get_अङ्कः_तन्तवः(kernel_env->shire_आवरणम्);

    यदि (thread_id < 0) {
        प्रत्यागम 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    ध्रुव भिन्न * src0_दत्त = (ध्रुव भिन्न *) src0->दत्त;
    भिन्न *       dst_दत्त  = (भिन्न *) dst->दत्त;

    यदि (!src0_दत्त || !dst_दत्त) {
        प्रत्यागम -1;
    }

    ध्रुव int32_t n_groups = params->n_groups;
    ध्रुव भिन्न   eps      = params->eps;

    यदि (n_groups <= 0 || eps < 0.0f) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t ne0 = dst->ne[0];
    ध्रुव int64_t ne1 = dst->ne[1];
    ध्रुव int64_t ne2 = dst->ne[2];
    ध्रुव int64_t ne3 = dst->ne[3];

    यदि (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t nb1  = dst->nb[1];
    ध्रुव int64_t nb2  = dst->nb[2];
    ध्रुव int64_t nb3  = dst->nb[3];
    ध्रुव int64_t nb01 = src0->nb[1];
    ध्रुव int64_t nb02 = src0->nb[2];
    ध्रुव int64_t nb03 = src0->nb[3];

    ध्रुव int64_t channels_प्रत्येकम्_गणः = (ne2 + n_groups - 1) / n_groups;
    यदि (channels_प्रत्येकम्_गणः <= 0) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t सक्रिय_groups        = (ne2 + channels_प्रत्येकम्_गणः - 1) / channels_प्रत्येकम्_गणः;
    ध्रुव int64_t योग_work           = सक्रिय_groups * ne3;
    ध्रुव int64_t पङ्क्तिकाः_प्रत्येकम्_write_गणः = et_पङ्क्तिकाः_प्रत्येकम्_cacheline_गणः(ne0, परिमाण(भिन्न));

    प्रति (int64_t work = thread_id; work < योग_work; work += अङ्कः_तन्तवः) {
        ध्रुव int64_t i3        = work / सक्रिय_groups;
        ध्रुव int64_t गणः_idx = work % सक्रिय_groups;

        ध्रुव int64_t channel_start = गणः_idx * channels_प्रत्येकम्_गणः;
        int64_t       channel_अन्तः   = channel_start + channels_प्रत्येकम्_गणः;
        यदि (channel_अन्तः > ne2) {
            channel_अन्तः = ne2;
        }

        ध्रुव int64_t channel_गणनम् = channel_अन्तः - channel_start;
        यदि (channel_गणनम् <= 0) {
            अनुवर्त;
        }

        भिन्न योग_2   = 0.0f;
        भिन्न denom = 0.0f;
        प्रति (int64_t i2 = channel_start; i2 < channel_अन्तः; ++i2) {
            प्रति (int64_t i1 = 0; i1 < ne1; ++i1) {
                ध्रुव भिन्न * src_row = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_दत्त + i3 * nb03 + i2 * nb02 + i1 * nb01);
                प्रति (int64_t i0 = 0; i0 < ne0; ++i0) {
                    योग_2 += src_row[i0];
                    denom += 1.0f;
                }
            }
        }

        ध्रुव भिन्न माध्यम्_2 = et_fdiv(योग_2, denom);

        भिन्न var_योग = 0.0f;
        प्रति (int64_t i2 = channel_start; i2 < channel_अन्तः; ++i2) {
            प्रति (int64_t i1 = 0; i1 < ne1; ++i1) {
                ध्रुव भिन्न * src_row = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_दत्त + i3 * nb03 + i2 * nb02 + i1 * nb01);
                प्रति (int64_t i0 = 0; i0 < ne0; ++i0) {
                    ध्रुव भिन्न centered = src_row[i0] - माध्यम्_2;
                    var_योग += centered * centered;
                }
            }
        }

        ध्रुव भिन्न variance = et_fdiv(var_योग, denom);
        ध्रुव भिन्न scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        यदि (ne0 % 16 == 0) {
            प्रति (int64_t i2 = channel_start; i2 < channel_अन्तः; ++i2) {
                प्रति (int64_t i1 = 0; i1 < ne1; ++i1) {
                    ध्रुव भिन्न * src_row =
                        (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_दत्त + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    भिन्न * dst_row = (भिन्न *) ((वर्ण *) dst_दत्त + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    प्रति (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - माध्यम्_2) * scale;
                    }
                }
            }
        } अन्यथा {
            ध्रुव int64_t योग_पङ्क्तिकाः_अन्तर्_गणः = channel_गणनम् * ne1;
            ध्रुव int64_t योग_write_groups  = (योग_पङ्क्तिकाः_अन्तर्_गणः + पङ्क्तिकाः_प्रत्येकम्_write_गणः - 1) / पङ्क्तिकाः_प्रत्येकम्_write_गणः;

            प्रति (int64_t write_गणः = 0; write_गणः < योग_write_groups; ++write_गणः) {
                ध्रुव int64_t row_start = write_गणः * पङ्क्तिकाः_प्रत्येकम्_write_गणः;
                int64_t       row_अन्तः   = row_start + पङ्क्तिकाः_प्रत्येकम्_write_गणः;
                यदि (row_अन्तः > योग_पङ्क्तिकाः_अन्तर्_गणः) {
                    row_अन्तः = योग_पङ्क्तिकाः_अन्तर्_गणः;
                }

                प्रति (int64_t row = row_start; row < row_अन्तः; ++row) {
                    ध्रुव int64_t local_i2 = row / ne1;
                    ध्रुव int64_t i1       = row % ne1;
                    ध्रुव int64_t i2       = channel_start + local_i2;

                    ध्रुव भिन्न * src_row =
                        (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_दत्त + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    भिन्न * dst_row = (भिन्न *) ((वर्ण *) dst_दत्त + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    प्रति (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_row[i0] = (src_row[i0] - माध्यम्_2) * scale;
                    }
                }
            }
        }
    }

    प्रत्यागम 0;
}
