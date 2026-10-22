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

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_سيٽ_params {
    بنية ggml_tensor src1;
    بنية ggml_tensor dst;
    int32_t            nb1;
    int32_t            nb2;
    int32_t            nb3;
    int32_t            وٿي;
};

ساكن مضمن فراغ copy_صف_aligned(عائم * dst, ثابت عائم * src, int32_t n) {
    لكل (int32_t i = 0; i < n; i += 8) {
        __asm__ متطاير(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(عائم (*)[8]) & dst[i])
            : [src_vec] "m"(*(ثابت عائم (*)[8]) & src[i])
            : "f11");
    }
}

صحيح داخلا_نقطو(بنية ggml_et_سيٽ_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح انگ_ڌاڳا = get_انگ_ڌاڳا(kernel_env->shire_نقاب);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    بنية ggml_tensor * src1 = &params->src1;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    ثابت عائم * src1_معلومات = (ثابت عائم *) src1->معلومات;
    عائم *       dst_معلومات  = (عائم *) dst->معلومات;
    إذا (!src1_معلومات || !dst_معلومات) {
        أرجع -1;
    }

    ثابت int64_t ne10 = src1->ne[0];
    ثابت int64_t ne11 = src1->ne[1];
    ثابت int64_t ne12 = src1->ne[2];
    ثابت int64_t ne13 = src1->ne[3];

    إذا (src1->nb[0] != حجم(عائم) || dst->nb[0] != حجم(عائم) || ne10 % 16 != 0) {
        أرجع -1;
    }

    ثابت int64_t nb11 = src1->nb[1];
    ثابت int64_t nb12 = src1->nb[2];
    ثابت int64_t nb13 = src1->nb[3];

    ثابت int64_t dnb1   = params->nb1;
    ثابت int64_t dnb2   = params->nb2;
    ثابت int64_t dnb3   = params->nb3;
    ثابت int64_t وٿي = params->وٿي;

    ثابت int64_t ڪل_صفون = ne11 * ne12 * ne13;

    لكل (int64_t صف = thread_id; صف < ڪل_صفون; صف += انگ_ڌاڳا) {
        ثابت int64_t i1 = صف % ne11;
        ثابت int64_t i2 = (صف / ne11) % ne12;
        ثابت int64_t i3 = صف / (ne11 * ne12);

        ثابت عائم * src_صف = (ثابت عائم *) ((ثابت حرف *) src1_معلومات + i1 * nb11 + i2 * nb12 + i3 * nb13);
        عائم *       dst_صف = (عائم *) ((حرف *) dst_معلومات + وٿي + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        copy_صف_aligned(dst_صف, src_صف, (int32_t) ne10);
    }

    أرجع 0;
}
