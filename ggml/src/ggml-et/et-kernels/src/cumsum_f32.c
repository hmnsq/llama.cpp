//******************************************************************************
// CUMSUM F32 Kernel
// Computes an inclusive prefix sum along dim 0 for each row in higher dims.
// First-pass implementation: scalar and row-contiguous input/output only.
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_cumsum_params {
    بنية ggml_tensor src0;
    بنية ggml_tensor dst;
};

صحيح entry_point(بنية ggml_et_cumsum_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحيح رەقەم_threads = get_رەقەم_threads(kernel_env->shire_mask);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    عائم * src0_data = (عائم *) src0->data;
    عائم * dst_data  = (عائم *) dst->data;

    إذا (!src0_data || !dst_data) {
        أرجع -1;
    }

    ثابت int64_t ne0 = src0->ne[0];
    ثابت int64_t ne1 = src0->ne[1];
    ثابت int64_t ne2 = src0->ne[2];
    ثابت int64_t ne3 = src0->ne[3];

    ثابت size_t snb0 = src0->nb[0];
    ثابت size_t snb1 = src0->nb[1];
    ثابت size_t snb2 = src0->nb[2];
    ثابت size_t snb3 = src0->nb[3];

    ثابت size_t dnb0 = dst->nb[0];
    ثابت size_t dnb1 = dst->nb[1];
    ثابت size_t dnb2 = dst->nb[2];
    ثابت size_t dnb3 = dst->nb[3];

    إذا (snb0 != حجم(عائم) || dnb0 != حجم(عائم)) {
        أرجع -1;
    }

    ثابت int64_t total_rows     = ne1 * ne2 * ne3;
    ثابت int64_t rows_per_group = et_rows_per_cacheline_group(ne0, حجم(عائم));
    ثابت int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

    لكل (int64_t grp = thread_id; grp < total_groups; grp += رەقەم_threads) {
        ثابت int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        إذا (row_end > total_rows) {
            row_end = total_rows;
        }

        لكل (int64_t row = row_start; row < row_end; ++row) {
            int64_t i1 = row % ne1;
            int64_t i2 = (row / ne1) % ne2;
            int64_t i3 = row / (ne1 * ne2);

            ثابت عائم * src_row = (ثابت عائم *) ((ثابت حرف *) src0_data + i1 * snb1 + i2 * snb2 + i3 * snb3);
            عائم *       dst_row = (عائم *) ((حرف *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            عائم acc = 0.0f;
            لكل (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_row[i0];
                dst_row[i0] = acc;
            }
        }
    }

    أرجع 0;
}
