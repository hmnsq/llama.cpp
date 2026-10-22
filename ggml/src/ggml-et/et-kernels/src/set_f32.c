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

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <stdint.h>

கட்டமைப்பு ggml_et_set_params {
    கட்டமைப்பு ggml_tensor src1;
    கட்டமைப்பு ggml_tensor dst;
    int32_t            nb1;
    int32_t            nb2;
    int32_t            nb3;
    int32_t            offset;
};

நிலையான உள்வரி வெறுமை copy_row_aligned(தசம * dst, மாறிலி தசம * src, int32_t n) {
    க்கு (int32_t i = 0; i < n; i += 8) {
        __asm__ நிலையற்ற(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(தசம (*)[8]) & dst[i])
            : [src_vec] "m"(*(மாறிலி தசம (*)[8]) & src[i])
            : "f11");
    }
}

முழு entry_point(கட்டமைப்பு ggml_et_set_params * params, வெறுமை * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    எனில் (!kernel_env) {
        திரும்பு -1;
    }

    முழு thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    முழு இலக்கம்_threads = get_இலக்கம்_threads(kernel_env->shire_mask);

    எனில் (thread_id < 0) {
        திரும்பு 0;
    }

    எனில் (params == 0 || ((uint64_t) params & 0x7) != 0) {
        திரும்பு -1;
    }

    கட்டமைப்பு ggml_tensor * src1 = &params->src1;
    கட்டமைப்பு ggml_tensor * dst  = &params->dst;

    எனில் (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        திரும்பு -1;
    }

    மாறிலி தசம * src1_data = (மாறிலி தசம *) src1->data;
    தசம *       dst_data  = (தசம *) dst->data;
    எனில் (!src1_data || !dst_data) {
        திரும்பு -1;
    }

    மாறிலி int64_t ne10 = src1->ne[0];
    மாறிலி int64_t ne11 = src1->ne[1];
    மாறிலி int64_t ne12 = src1->ne[2];
    மாறிலி int64_t ne13 = src1->ne[3];

    எனில் (src1->nb[0] != அளவு(தசம) || dst->nb[0] != அளவு(தசம) || ne10 % 16 != 0) {
        திரும்பு -1;
    }

    மாறிலி int64_t nb11 = src1->nb[1];
    மாறிலி int64_t nb12 = src1->nb[2];
    மாறிலி int64_t nb13 = src1->nb[3];

    மாறிலி int64_t dnb1   = params->nb1;
    மாறிலி int64_t dnb2   = params->nb2;
    மாறிலி int64_t dnb3   = params->nb3;
    மாறிலி int64_t offset = params->offset;

    மாறிலி int64_t total_rows = ne11 * ne12 * ne13;

    க்கு (int64_t row = thread_id; row < total_rows; row += இலக்கம்_threads) {
        மாறிலி int64_t i1 = row % ne11;
        மாறிலி int64_t i2 = (row / ne11) % ne12;
        மாறிலி int64_t i3 = row / (ne11 * ne12);

        மாறிலி தசம * src_row = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src1_data + i1 * nb11 + i2 * nb12 + i3 * nb13);
        தசம *       dst_row = (தசம *) ((எழுத்து *) dst_data + offset + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        copy_row_aligned(dst_row, src_row, (int32_t) ne10);
    }

    திரும்பு 0;
}
