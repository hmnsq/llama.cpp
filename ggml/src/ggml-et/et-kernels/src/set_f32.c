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

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ಸಂರಚನಾ ggml_et_set_params {
    ಸಂರಚನಾ ggml_tensor src1;
    ಸಂರಚನಾ ggml_tensor dst;
    int32_t            nb1;
    int32_t            nb2;
    int32_t            nb3;
    int32_t            offset;
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ copy_row_aligned(ಭಿನ್ನ * dst, ಧ್ರುವ ಭಿನ್ನ * src, int32_t n) {
    ಪ್ರತಿ (int32_t i = 0; i < n; i += 8) {
        __asm__ ಚಞ್ಚಲ(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(ಭಿನ್ನ (*)[8]) & dst[i])
            : [src_vec] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & src[i])
            : "f11");
    }
}

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_set_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಯದಿ (!kernel_env) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪೂರ್ಣಾಙ್ಕ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_threads = get_ಅಂಕೆ_threads(kernel_env->shire_mask);

    ಯದಿ (thread_id < 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಯದಿ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಸಂರಚನಾ ggml_tensor * src1 = &params->src1;
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;

    ಯದಿ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ ಭಿನ್ನ * src1_data = (ಧ್ರುವ ಭಿನ್ನ *) src1->data;
    ಭಿನ್ನ *       dst_data  = (ಭಿನ್ನ *) dst->data;
    ಯದಿ (!src1_data || !dst_data) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ int64_t ne10 = src1->ne[0];
    ಧ್ರುವ int64_t ne11 = src1->ne[1];
    ಧ್ರುವ int64_t ne12 = src1->ne[2];
    ಧ್ರುವ int64_t ne13 = src1->ne[3];

    ಯದಿ (src1->nb[0] != ಪರಿಮಾಣ(ಭಿನ್ನ) || dst->nb[0] != ಪರಿಮಾಣ(ಭಿನ್ನ) || ne10 % 16 != 0) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ int64_t nb11 = src1->nb[1];
    ಧ್ರುವ int64_t nb12 = src1->nb[2];
    ಧ್ರುವ int64_t nb13 = src1->nb[3];

    ಧ್ರುವ int64_t dnb1   = params->nb1;
    ಧ್ರುವ int64_t dnb2   = params->nb2;
    ಧ್ರುವ int64_t dnb3   = params->nb3;
    ಧ್ರುವ int64_t offset = params->offset;

    ಧ್ರುವ int64_t total_rows = ne11 * ne12 * ne13;

    ಪ್ರತಿ (int64_t row = thread_id; row < total_rows; row += ಅಂಕೆ_threads) {
        ಧ್ರುವ int64_t i1 = row % ne11;
        ಧ್ರುವ int64_t i2 = (row / ne11) % ne12;
        ಧ್ರುವ int64_t i3 = row / (ne11 * ne12);

        ಧ್ರುವ ಭಿನ್ನ * src_row = (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ ವರ್ಣ *) src1_data + i1 * nb11 + i2 * nb12 + i3 * nb13);
        ಭಿನ್ನ *       dst_row = (ಭಿನ್ನ *) ((ವರ್ಣ *) dst_data + offset + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        copy_row_aligned(dst_row, src_row, (int32_t) ne10);
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}
