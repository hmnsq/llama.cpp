//******************************************************************************
// CUMSUM F32 Kernel
// Computes an inclusive prefix sum along dim 0 for each row in higher dims.
// First-pass implementation: scalar and row-contiguous input/output only.
//******************************************************************************

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

কাঠামো ggml_et_cumsum_params {
    কাঠামো ggml_tensor src0;
    কাঠামো ggml_tensor dst;
};

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_cumsum_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ফেরত -1;
    }

    পূর্ণসংখ্যা thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণসংখ্যা অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ফেরত 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ফেরত -1;
    }

    কাঠামো ggml_tensor * src0 = &params->src0;
    কাঠামো ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ফেরত -1;
    }

    দশমিক * src0_তথ্য = (দশমিক *) src0->তথ্য;
    দশমিক * dst_তথ্য  = (দশমিক *) dst->তথ্য;

    যদি (!src0_তথ্য || !dst_তথ্য) {
        ফেরত -1;
    }

    ধ্রুবক int64_t ne0 = src0->ne[0];
    ধ্রুবক int64_t ne1 = src0->ne[1];
    ধ্রুবক int64_t ne2 = src0->ne[2];
    ধ্রুবক int64_t ne3 = src0->ne[3];

    ধ্রুবক size_t snb0 = src0->nb[0];
    ধ্রুবক size_t snb1 = src0->nb[1];
    ধ্রুবক size_t snb2 = src0->nb[2];
    ধ্রুবক size_t snb3 = src0->nb[3];

    ধ্রুবক size_t dnb0 = dst->nb[0];
    ধ্রুবক size_t dnb1 = dst->nb[1];
    ধ্রুবক size_t dnb2 = dst->nb[2];
    ধ্রুবক size_t dnb3 = dst->nb[3];

    যদি (snb0 != আকার(দশমিক) || dnb0 != আকার(দশমিক)) {
        ফেরত -1;
    }

    ধ্রুবক int64_t যোগ_rows     = ne1 * ne2 * ne3;
    ধ্রুবক int64_t rows_per_group = et_rows_per_cacheline_group(ne0, আকার(দশমিক));
    ধ্রুবক int64_t যোগ_groups   = (যোগ_rows + rows_per_group - 1) / rows_per_group;

    জন্য (int64_t grp = thread_id; grp < যোগ_groups; grp += অঙ্ক_threads) {
        ধ্রুবক int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        যদি (row_end > যোগ_rows) {
            row_end = যোগ_rows;
        }

        জন্য (int64_t row = row_start; row < row_end; ++row) {
            int64_t i1 = row % ne1;
            int64_t i2 = (row / ne1) % ne2;
            int64_t i3 = row / (ne1 * ne2);

            ধ্রুবক দশমিক * src_row = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src0_তথ্য + i1 * snb1 + i2 * snb2 + i3 * snb3);
            দশমিক *       dst_row = (দশমিক *) ((অক্ষর *) dst_তথ্য + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            দশমিক acc = 0.0f;
            জন্য (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_row[i0];
                dst_row[i0] = acc;
            }
        }
    }

    ফেরত 0;
}
