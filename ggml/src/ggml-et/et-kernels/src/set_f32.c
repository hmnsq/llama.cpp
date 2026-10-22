//******************************************************************************
// SET F32 Kernel
// Minimal ET implementation for inplace F32 SET into a contiguous destination
// using a contiguous F32 source view and explicit destination view strides.
//
// Supported shape family:
// - dst/base is contiguous F32
// - src1 is contiguous F32
// - src1.ne[0] is cacheline-aligned (multiple of 16 floats)
// - destination view strides/offset are cacheline-aligned
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdint.h>

ساختار ggml_et_set_params {
    ساختار ggml_tensor src1;
    ساختار ggml_tensor dst;
    int32_t            nb1;
    int32_t            nb2;
    int32_t            nb3;
    int32_t            offset;
};

ایستا درونخطی تهی copy_row_aligned(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    برای (int32_t i = 0; i < n; i += 8) {
        __asm__ فرار(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(اعشاری (*)[8]) & dst[i])
            : [src_vec] "m"(*(ثابت اعشاری (*)[8]) & src[i])
            : "f11");
    }
}

صحیح entry_point(ساختار ggml_et_set_params * params, تهی * env) {
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

    ساختار ggml_tensor * src1 = &params->src1;
    ساختار ggml_tensor * dst  = &params->dst;

    اگر (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        بازگشت -1;
    }

    ثابت اعشاری * src1_data = (ثابت اعشاری *) src1->data;
    اعشاری *       dst_data  = (اعشاری *) dst->data;
    اگر (!src1_data || !dst_data) {
        بازگشت -1;
    }

    ثابت int64_t ne10 = src1->ne[0];
    ثابت int64_t ne11 = src1->ne[1];
    ثابت int64_t ne12 = src1->ne[2];
    ثابت int64_t ne13 = src1->ne[3];

    اگر (src1->nb[0] != اندازه(اعشاری) || dst->nb[0] != اندازه(اعشاری) || ne10 % 16 != 0) {
        بازگشت -1;
    }

    ثابت int64_t nb11 = src1->nb[1];
    ثابت int64_t nb12 = src1->nb[2];
    ثابت int64_t nb13 = src1->nb[3];

    ثابت int64_t dnb1   = params->nb1;
    ثابت int64_t dnb2   = params->nb2;
    ثابت int64_t dnb3   = params->nb3;
    ثابت int64_t offset = params->offset;

    ثابت int64_t total_rows = ne11 * ne12 * ne13;

    برای (int64_t row = thread_id; row < total_rows; row += رقم_threads) {
        ثابت int64_t i1 = row % ne11;
        ثابت int64_t i2 = (row / ne11) % ne12;
        ثابت int64_t i3 = row / (ne11 * ne12);

        ثابت اعشاری * src_row = (ثابت اعشاری *) ((ثابت نویسه *) src1_data + i1 * nb11 + i2 * nb12 + i3 * nb13);
        اعشاری *       dst_row = (اعشاری *) ((نویسه *) dst_data + offset + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        copy_row_aligned(dst_row, src_row, (int32_t) ne10);
    }

    بازگشت 0;
}
