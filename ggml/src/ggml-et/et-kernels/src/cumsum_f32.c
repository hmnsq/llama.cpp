//******************************************************************************
// CUMSUM F32 Kernel
// Computes an inclusive prefix sum along dim 0 for each row in higher dims.
// First-pass implementation: scalar and row-contiguous input/output only.
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ಸಂರಚನಾ ggml_et_cumsum_params {
    ಸಂರಚನಾ ggml_tensor src0;
    ಸಂರಚನಾ ggml_tensor dst;
};

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_cumsum_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಯದಿ (!kernel_env) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪೂರ್ಣಾಙ್ಕ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_threads = get_ಅಂಕೆ_threads(kernel_env->shire_mask);

    ಯದಿ (thread_id < 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಯದಿ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಸಂರಚನಾ ggml_tensor * src0 = &params->src0;
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;

    ಯದಿ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಭಿನ್ನ * src0_data = (ಭಿನ್ನ *) src0->data;
    ಭಿನ್ನ * dst_data  = (ಭಿನ್ನ *) dst->data;

    ಯದಿ (!src0_data || !dst_data) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ int64_t ne0 = src0->ne[0];
    ಧ್ರುವ int64_t ne1 = src0->ne[1];
    ಧ್ರುವ int64_t ne2 = src0->ne[2];
    ಧ್ರುವ int64_t ne3 = src0->ne[3];

    ಧ್ರುವ size_t snb0 = src0->nb[0];
    ಧ್ರುವ size_t snb1 = src0->nb[1];
    ಧ್ರುವ size_t snb2 = src0->nb[2];
    ಧ್ರುವ size_t snb3 = src0->nb[3];

    ಧ್ರುವ size_t dnb0 = dst->nb[0];
    ಧ್ರುವ size_t dnb1 = dst->nb[1];
    ಧ್ರುವ size_t dnb2 = dst->nb[2];
    ಧ್ರುವ size_t dnb3 = dst->nb[3];

    ಯದಿ (snb0 != ಪರಿಮಾಣ(ಭಿನ್ನ) || dnb0 != ಪರಿಮಾಣ(ಭಿನ್ನ)) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ int64_t total_rows     = ne1 * ne2 * ne3;
    ಧ್ರುವ int64_t rows_per_group = et_rows_per_cacheline_group(ne0, ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಧ್ರುವ int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

    ಪ್ರತಿ (int64_t grp = thread_id; grp < total_groups; grp += ಅಂಕೆ_threads) {
        ಧ್ರುವ int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        ಯದಿ (row_end > total_rows) {
            row_end = total_rows;
        }

        ಪ್ರತಿ (int64_t row = row_start; row < row_end; ++row) {
            int64_t i1 = row % ne1;
            int64_t i2 = (row / ne1) % ne2;
            int64_t i3 = row / (ne1 * ne2);

            ಧ್ರುವ ಭಿನ್ನ * src_row = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src0_data + i1 * snb1 + i2 * snb2 + i3 * snb3);
            ಭಿನ್ನ *       dst_row = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            ಭಿನ್ನ acc = 0.0f;
            ಪ್ರತಿ (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_row[i0];
                dst_row[i0] = acc;
            }
        }
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}
