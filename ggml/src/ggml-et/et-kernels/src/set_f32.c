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

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>

ସଂରଚନା ggml_et_set_params {
    ସଂରଚନା ggml_tensor src1;
    ସଂରଚନା ggml_tensor dst;
    int32_t            nb1;
    int32_t            nb2;
    int32_t            nb3;
    int32_t            offset;
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ copy_row_aligned(ଦଶମିକ * dst, ଅଚଳ ଦଶମିକ * src, int32_t n) {
    ପାଇଁ (int32_t i = 0; i < n; i += 8) {
        __asm__ ଅସ୍ଥିର(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(ଦଶମିକ (*)[8]) & dst[i])
            : [src_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & src[i])
            : "f11");
    }
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_set_params * params, ଶୂନ୍ୟ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ଯଦି (!kernel_env) {
        ଫେରନ୍ତୁ -1;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_threads = get_ଅଙ୍କ_threads(kernel_env->shire_mask);

    ଯଦି (thread_id < 0) {
        ଫେରନ୍ତୁ 0;
    }

    ଯଦି (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ଫେରନ୍ତୁ -1;
    }

    ସଂରଚନା ggml_tensor * src1 = &params->src1;
    ସଂରଚନା ggml_tensor * dst  = &params->dst;

    ଯଦି (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ ଦଶମିକ * src1_data = (ଅଚଳ ଦଶମିକ *) src1->data;
    ଦଶମିକ *       dst_data  = (ଦଶମିକ *) dst->data;
    ଯଦି (!src1_data || !dst_data) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int64_t ne10 = src1->ne[0];
    ଅଚଳ int64_t ne11 = src1->ne[1];
    ଅଚଳ int64_t ne12 = src1->ne[2];
    ଅଚଳ int64_t ne13 = src1->ne[3];

    ଯଦି (src1->nb[0] != ଆକାର(ଦଶମିକ) || dst->nb[0] != ଆକାର(ଦଶମିକ) || ne10 % 16 != 0) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int64_t nb11 = src1->nb[1];
    ଅଚଳ int64_t nb12 = src1->nb[2];
    ଅଚଳ int64_t nb13 = src1->nb[3];

    ଅଚଳ int64_t dnb1   = params->nb1;
    ଅଚଳ int64_t dnb2   = params->nb2;
    ଅଚଳ int64_t dnb3   = params->nb3;
    ଅଚଳ int64_t offset = params->offset;

    ଅଚଳ int64_t total_rows = ne11 * ne12 * ne13;

    ପାଇଁ (int64_t row = thread_id; row < total_rows; row += ଅଙ୍କ_threads) {
        ଅଚଳ int64_t i1 = row % ne11;
        ଅଚଳ int64_t i2 = (row / ne11) % ne12;
        ଅଚଳ int64_t i3 = row / (ne11 * ne12);

        ଅଚଳ ଦଶମିକ * src_row = (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src1_data + i1 * nb11 + i2 * nb12 + i3 * nb13);
        ଦଶମିକ *       dst_row = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst_data + offset + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        copy_row_aligned(dst_row, src_row, (int32_t) ne10);
    }

    ଫେରନ୍ତୁ 0;
}
