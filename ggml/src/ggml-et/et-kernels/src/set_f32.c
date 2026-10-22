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

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <stdint.h>

સંરચના ggml_et_set_params {
    સંરચના ggml_tensor src1;
    સંરચના ggml_tensor dst;
    int32_t            nb1;
    int32_t            nb2;
    int32_t            nb3;
    int32_t            offset;
};

સ્થિર અંતઃરેખ શૂન્ય copy_row_aligned(દશાંશ * dst, અચલ દશાંશ * src, int32_t n) {
    માટે (int32_t i = 0; i < n; i += 8) {
        __asm__ અસ્થિર(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(દશાંશ (*)[8]) & dst[i])
            : [src_vec] "m"(*(અચલ દશાંશ (*)[8]) & src[i])
            : "f11");
    }
}

પૂર્ણાંક entry_point(સંરચના ggml_et_set_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    જો (!kernel_env) {
        પાછા -1;
    }

    પૂર્ણાંક thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    પૂર્ણાંક અંક_threads = get_અંક_threads(kernel_env->shire_mask);

    જો (thread_id < 0) {
        પાછા 0;
    }

    જો (params == 0 || ((uint64_t) params & 0x7) != 0) {
        પાછા -1;
    }

    સંરચના ggml_tensor * src1 = &params->src1;
    સંરચના ggml_tensor * dst  = &params->dst;

    જો (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા -1;
    }

    અચલ દશાંશ * src1_data = (અચલ દશાંશ *) src1->data;
    દશાંશ *       dst_data  = (દશાંશ *) dst->data;
    જો (!src1_data || !dst_data) {
        પાછા -1;
    }

    અચલ int64_t ne10 = src1->ne[0];
    અચલ int64_t ne11 = src1->ne[1];
    અચલ int64_t ne12 = src1->ne[2];
    અચલ int64_t ne13 = src1->ne[3];

    જો (src1->nb[0] != કદ(દશાંશ) || dst->nb[0] != કદ(દશાંશ) || ne10 % 16 != 0) {
        પાછા -1;
    }

    અચલ int64_t nb11 = src1->nb[1];
    અચલ int64_t nb12 = src1->nb[2];
    અચલ int64_t nb13 = src1->nb[3];

    અચલ int64_t dnb1   = params->nb1;
    અચલ int64_t dnb2   = params->nb2;
    અચલ int64_t dnb3   = params->nb3;
    અચલ int64_t offset = params->offset;

    અચલ int64_t total_rows = ne11 * ne12 * ne13;

    માટે (int64_t row = thread_id; row < total_rows; row += અંક_threads) {
        અચલ int64_t i1 = row % ne11;
        અચલ int64_t i2 = (row / ne11) % ne12;
        અચલ int64_t i3 = row / (ne11 * ne12);

        અચલ દશાંશ * src_row = (અચલ દશાંશ *) ((અચલ અક્ષર *) src1_data + i1 * nb11 + i2 * nb12 + i3 * nb13);
        દશાંશ *       dst_row = (દશાંશ *) ((અક્ષર *) dst_data + offset + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        copy_row_aligned(dst_row, src_row, (int32_t) ne10);
    }

    પાછા 0;
}
