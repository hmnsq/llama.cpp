//******************************************************************************
// CUMSUM F32 Kernel
// Computes an inclusive prefix sum along dim 0 for each row in higher dims.
// First-pass implementation: scalar and row-contiguous input/output only.
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdint.h>

ساختار ggml_et_cumsum_params {
    ساختار ggml_tensor src0;
    ساختار ggml_tensor dst;
};

صحیح entry_point(ساختار ggml_et_cumsum_params * params, تهی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        بازگشت -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحیح رقم_threads = get_رقم_threads(kernel_env->shire_mask);

    اگر (thread_id < 0) {
        بازگشت 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        بازگشت -1;
    }

    ساختار ggml_tensor * src0 = &params->src0;
    ساختار ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        بازگشت -1;
    }

    اعشاری * src0_data = (اعشاری *) src0->data;
    اعشاری * dst_data  = (اعشاری *) dst->data;

    اگر (!src0_data || !dst_data) {
        بازگشت -1;
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

    اگر (snb0 != اندازه(اعشاری) || dnb0 != اندازه(اعشاری)) {
        بازگشت -1;
    }

    ثابت int64_t total_rows     = ne1 * ne2 * ne3;
    ثابت int64_t rows_per_group = et_rows_per_cacheline_group(ne0, اندازه(اعشاری));
    ثابت int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

    برای (int64_t grp = thread_id; grp < total_groups; grp += رقم_threads) {
        ثابت int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        اگر (row_end > total_rows) {
            row_end = total_rows;
        }

        برای (int64_t row = row_start; row < row_end; ++row) {
            int64_t i1 = row % ne1;
            int64_t i2 = (row / ne1) % ne2;
            int64_t i3 = row / (ne1 * ne2);

            ثابت اعشاری * src_row = (ثابت اعشاری *) ((ثابت نویسه *) src0_data + i1 * snb1 + i2 * snb2 + i3 * snb3);
            اعشاری *       dst_row = (اعشاری *) ((نویسه *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            اعشاری acc = 0.0f;
            برای (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_row[i0];
                dst_row[i0] = acc;
            }
        }
    }

    بازگشت 0;
}
