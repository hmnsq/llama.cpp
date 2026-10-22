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

ساخت ggml_et_سیٹ_params {
    ساخت ggml_tensor src1;
    ساخت ggml_tensor dst;
    int32_t            nb1;
    int32_t            nb2;
    int32_t            nb3;
    int32_t            انحراف;
};

ساکن اندرونی خالی copy_صف_aligned(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    برائے (int32_t i = 0; i < n; i += 8) {
        __asm__ متلون(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(اعشاری (*)[8]) & dst[i])
            : [src_vec] "m"(*(ثابت اعشاری (*)[8]) & src[i])
            : "f11");
    }
}

صحیح اندراج_نقطہ(ساخت ggml_et_سیٹ_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        واپس -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    اگر (thread_id < 0) {
        واپس 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        واپس -1;
    }

    ساخت ggml_tensor * src1 = &params->src1;
    ساخت ggml_tensor * dst  = &params->dst;

    اگر (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس -1;
    }

    ثابت اعشاری * src1_ڈیٹا = (ثابت اعشاری *) src1->ڈیٹا;
    اعشاری *       dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;
    اگر (!src1_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;
    }

    ثابت int64_t ne10 = src1->ne[0];
    ثابت int64_t ne11 = src1->ne[1];
    ثابت int64_t ne12 = src1->ne[2];
    ثابت int64_t ne13 = src1->ne[3];

    اگر (src1->nb[0] != حجم(اعشاری) || dst->nb[0] != حجم(اعشاری) || ne10 % 16 != 0) {
        واپس -1;
    }

    ثابت int64_t nb11 = src1->nb[1];
    ثابت int64_t nb12 = src1->nb[2];
    ثابت int64_t nb13 = src1->nb[3];

    ثابت int64_t dnb1   = params->nb1;
    ثابت int64_t dnb2   = params->nb2;
    ثابت int64_t dnb3   = params->nb3;
    ثابت int64_t انحراف = params->انحراف;

    ثابت int64_t کل_صفیں = ne11 * ne12 * ne13;

    برائے (int64_t صف = thread_id; صف < کل_صفیں; صف += عدد_دھاگے) {
        ثابت int64_t i1 = صف % ne11;
        ثابت int64_t i2 = (صف / ne11) % ne12;
        ثابت int64_t i3 = صف / (ne11 * ne12);

        ثابت اعشاری * src_صف = (ثابت اعشاری *) ((ثابت حرف *) src1_ڈیٹا + i1 * nb11 + i2 * nb12 + i3 * nb13);
        اعشاری *       dst_صف = (اعشاری *) ((حرف *) dst_ڈیٹا + انحراف + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        copy_صف_aligned(dst_صف, src_صف, (int32_t) ne10);
    }

    واپس 0;
}
