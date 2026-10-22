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

पूर्णाङ्क entry_point(संरचना ggml_et_cumsum_params * params, शून्य * env) {
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

    भिन्न * src0_data = (भिन्न *) src0->data;
    भिन्न * dst_data  = (भिन्न *) dst->data;

    यदि (!src0_data || !dst_data) {
        फर्काऊ -1;
    }

    अचर int64_t ne0 = src0->ne[0];
    अचर int64_t ne1 = src0->ne[1];
    अचर int64_t ne2 = src0->ne[2];
    अचर int64_t ne3 = src0->ne[3];

    अचर माप_प्रकार snb0 = src0->nb[0];
    अचर माप_प्रकार snb1 = src0->nb[1];
    अचर माप_प्रकार snb2 = src0->nb[2];
    अचर माप_प्रकार snb3 = src0->nb[3];

    अचर माप_प्रकार dnb0 = dst->nb[0];
    अचर माप_प्रकार dnb1 = dst->nb[1];
    अचर माप_प्रकार dnb2 = dst->nb[2];
    अचर माप_प्रकार dnb3 = dst->nb[3];

    यदि (snb0 != परिमाण(भिन्न) || dnb0 != परिमाण(भिन्न)) {
        फर्काऊ -1;
    }

    अचर int64_t total_rows     = ne1 * ne2 * ne3;
    अचर int64_t rows_per_group = et_rows_per_cacheline_group(ne0, परिमाण(भिन्न));
    अचर int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

    लागि (int64_t grp = thread_id; grp < total_groups; grp += अङ्क_threads) {
        अचर int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        यदि (row_end > total_rows) {
            row_end = total_rows;
        }

        लागि (int64_t row = row_start; row < row_end; ++row) {
            int64_t i1 = row % ne1;
            int64_t i2 = (row / ne1) % ne2;
            int64_t i3 = row / (ne1 * ne2);

            अचर भिन्न * src_row = (अचर भिन्न *) ((अचर वर्ण *) src0_data + i1 * snb1 + i2 * snb2 + i3 * snb3);
            भिन्न *       dst_row = (भिन्न *) ((वर्ण *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            भिन्न acc = 0.0f;
            लागि (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_row[i0];
                dst_row[i0] = acc;
            }
        }
    }

    फर्काऊ 0;
}
