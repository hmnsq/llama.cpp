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

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

संरचना ggml_et_set_params {
    संरचना ggml_tensor src1;
    संरचना ggml_tensor dst;
    int32_t            nb1;
    int32_t            nb2;
    int32_t            nb3;
    int32_t            offset;
};

स्थिर अन्तःस्थ शून्य copy_row_aligned(भिन्न * dst, ध्रुव भिन्न * src, int32_t n) {
    प्रति (int32_t i = 0; i < n; i += 8) {
        __asm__ चञ्चल(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(भिन्न (*)[8]) & dst[i])
            : [src_vec] "m"(*(ध्रुव भिन्न (*)[8]) & src[i])
            : "f11");
    }
}

पूर्णाङ्क entry_point(संरचना ggml_et_set_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    यदि (thread_id < 0) {
        प्रत्यागम 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;
    }

    संरचना ggml_tensor * src1 = &params->src1;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    ध्रुव भिन्न * src1_data = (ध्रुव भिन्न *) src1->data;
    भिन्न *       dst_data  = (भिन्न *) dst->data;
    यदि (!src1_data || !dst_data) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t ne10 = src1->ne[0];
    ध्रुव int64_t ne11 = src1->ne[1];
    ध्रुव int64_t ne12 = src1->ne[2];
    ध्रुव int64_t ne13 = src1->ne[3];

    यदि (src1->nb[0] != परिमाण(भिन्न) || dst->nb[0] != परिमाण(भिन्न) || ne10 % 16 != 0) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t nb11 = src1->nb[1];
    ध्रुव int64_t nb12 = src1->nb[2];
    ध्रुव int64_t nb13 = src1->nb[3];

    ध्रुव int64_t dnb1   = params->nb1;
    ध्रुव int64_t dnb2   = params->nb2;
    ध्रुव int64_t dnb3   = params->nb3;
    ध्रुव int64_t offset = params->offset;

    ध्रुव int64_t total_rows = ne11 * ne12 * ne13;

    प्रति (int64_t row = thread_id; row < total_rows; row += अंक_threads) {
        ध्रुव int64_t i1 = row % ne11;
        ध्रुव int64_t i2 = (row / ne11) % ne12;
        ध्रुव int64_t i3 = row / (ne11 * ne12);

        ध्रुव भिन्न * src_row = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src1_data + i1 * nb11 + i2 * nb12 + i3 * nb13);
        भिन्न *       dst_row = (भिन्न *) ((वर्ण *) dst_data + offset + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        copy_row_aligned(dst_row, src_row, (int32_t) ne10);
    }

    प्रत्यागम 0;
}
