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

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

কাঠামো ggml_et_set_params {
    কাঠামো ggml_tensor src1;
    কাঠামো ggml_tensor dst;
    int32_t            nb1;
    int32_t            nb2;
    int32_t            nb3;
    int32_t            offset;
};

স্থির অন্তঃরেখ শূন্য copy_row_aligned(দশমিক * dst, ধ্রুবক দশমিক * src, int32_t n) {
    জন্য (int32_t i = 0; i < n; i += 8) {
        __asm__ অস্থির(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(দশমিক (*)[8]) & dst[i])
            : [src_vec] "m"(*(ধ্রুবক দশমিক (*)[8]) & src[i])
            : "f11");
    }
}

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_set_params * params, শূন্য * env) {
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

    কাঠামো ggml_tensor * src1 = &params->src1;
    কাঠামো ggml_tensor * dst  = &params->dst;

    যদি (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ফেরত -1;
    }

    ধ্রুবক দশমিক * src1_তথ্য = (ধ্রুবক দশমিক *) src1->তথ্য;
    দশমিক *       dst_তথ্য  = (দশমিক *) dst->তথ্য;
    যদি (!src1_তথ্য || !dst_তথ্য) {
        ফেরত -1;
    }

    ধ্রুবক int64_t ne10 = src1->ne[0];
    ধ্রুবক int64_t ne11 = src1->ne[1];
    ধ্রুবক int64_t ne12 = src1->ne[2];
    ধ্রুবক int64_t ne13 = src1->ne[3];

    যদি (src1->nb[0] != আকার(দশমিক) || dst->nb[0] != আকার(দশমিক) || ne10 % 16 != 0) {
        ফেরত -1;
    }

    ধ্রুবক int64_t nb11 = src1->nb[1];
    ধ্রুবক int64_t nb12 = src1->nb[2];
    ধ্রুবক int64_t nb13 = src1->nb[3];

    ধ্রুবক int64_t dnb1   = params->nb1;
    ধ্রুবক int64_t dnb2   = params->nb2;
    ধ্রুবক int64_t dnb3   = params->nb3;
    ধ্রুবক int64_t offset = params->offset;

    ধ্রুবক int64_t যোগ_rows = ne11 * ne12 * ne13;

    জন্য (int64_t row = thread_id; row < যোগ_rows; row += অঙ্ক_threads) {
        ধ্রুবক int64_t i1 = row % ne11;
        ধ্রুবক int64_t i2 = (row / ne11) % ne12;
        ধ্রুবক int64_t i3 = row / (ne11 * ne12);

        ধ্রুবক দশমিক * src_row = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src1_তথ্য + i1 * nb11 + i2 * nb12 + i3 * nb13);
        দশমিক *       dst_row = (দশমিক *) ((অক্ষর *) dst_তথ্য + offset + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        copy_row_aligned(dst_row, src_row, (int32_t) ne10);
    }

    ফেরত 0;
}
