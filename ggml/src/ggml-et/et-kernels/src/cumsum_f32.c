//******************************************************************************
// CUMSUM F32 Kernel
// Computes an inclusive prefix sum along dim 0 for each row in higher dims.
// First-pass implementation: scalar and row-contiguous input/output only.
//******************************************************************************

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

संरचना ggml_et_cumsum_params {
    संरचना ggml_tensor src0;
    संरचना ggml_tensor dst;
};

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_cumsum_params * params, शून्य * env) {
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

    भिन्न * src0_दत्त = (भिन्न *) src0->दत्त;
    भिन्न * dst_दत्त  = (भिन्न *) dst->दत्त;

    यदि (!src0_दत्त || !dst_दत्त) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t ne0 = src0->ne[0];
    ध्रुव int64_t ne1 = src0->ne[1];
    ध्रुव int64_t ne2 = src0->ne[2];
    ध्रुव int64_t ne3 = src0->ne[3];

    ध्रुव size_t snb0 = src0->nb[0];
    ध्रुव size_t snb1 = src0->nb[1];
    ध्रुव size_t snb2 = src0->nb[2];
    ध्रुव size_t snb3 = src0->nb[3];

    ध्रुव size_t dnb0 = dst->nb[0];
    ध्रुव size_t dnb1 = dst->nb[1];
    ध्रुव size_t dnb2 = dst->nb[2];
    ध्रुव size_t dnb3 = dst->nb[3];

    यदि (snb0 != परिमाण(भिन्न) || dnb0 != परिमाण(भिन्न)) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t योग_पङ्क्तिकाः     = ne1 * ne2 * ne3;
    ध्रुव int64_t पङ्क्तिकाः_प्रत्येकम्_गणः = et_पङ्क्तिकाः_प्रत्येकम्_cacheline_गणः(ne0, परिमाण(भिन्न));
    ध्रुव int64_t योग_groups   = (योग_पङ्क्तिकाः + पङ्क्तिकाः_प्रत्येकम्_गणः - 1) / पङ्क्तिकाः_प्रत्येकम्_गणः;

    प्रति (int64_t grp = thread_id; grp < योग_groups; grp += अङ्कः_तन्तवः) {
        ध्रुव int64_t row_start = grp * पङ्क्तिकाः_प्रत्येकम्_गणः;
        int64_t       row_अन्तः   = row_start + पङ्क्तिकाः_प्रत्येकम्_गणः;
        यदि (row_अन्तः > योग_पङ्क्तिकाः) {
            row_अन्तः = योग_पङ्क्तिकाः;
        }

        प्रति (int64_t row = row_start; row < row_अन्तः; ++row) {
            int64_t i1 = row % ne1;
            int64_t i2 = (row / ne1) % ne2;
            int64_t i3 = row / (ne1 * ne2);

            ध्रुव भिन्न * src_row = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_दत्त + i1 * snb1 + i2 * snb2 + i3 * snb3);
            भिन्न *       dst_row = (भिन्न *) ((वर्ण *) dst_दत्त + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            भिन्न acc = 0.0f;
            प्रति (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_row[i0];
                dst_row[i0] = acc;
            }
        }
    }

    प्रत्यागम 0;
}
