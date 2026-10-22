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

स्थिर ओळीत शून्य copy_row_aligned(दशांश * dst, स्थिरांक दशांश * src, int32_t n) {
    साठी (int32_t i = 0; i < n; i += 8) {
        __asm__ अस्थिर(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(दशांश (*)[8]) & dst[i])
            : [src_vec] "m"(*(स्थिरांक दशांश (*)[8]) & src[i])
            : "f11");
    }
}

पूर्णांक entry_point(संरचना ggml_et_set_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जर (!kernel_env) {
        परत -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    जर (thread_id < 0) {
        परत 0;
    }

    जर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        परत -1;
    }

    संरचना ggml_tensor * src1 = &params->src1;
    संरचना ggml_tensor * dst  = &params->dst;

    जर (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        परत -1;
    }

    स्थिरांक दशांश * src1_data = (स्थिरांक दशांश *) src1->data;
    दशांश *       dst_data  = (दशांश *) dst->data;
    जर (!src1_data || !dst_data) {
        परत -1;
    }

    स्थिरांक int64_t ne10 = src1->ne[0];
    स्थिरांक int64_t ne11 = src1->ne[1];
    स्थिरांक int64_t ne12 = src1->ne[2];
    स्थिरांक int64_t ne13 = src1->ne[3];

    जर (src1->nb[0] != आकार(दशांश) || dst->nb[0] != आकार(दशांश) || ne10 % 16 != 0) {
        परत -1;
    }

    स्थिरांक int64_t nb11 = src1->nb[1];
    स्थिरांक int64_t nb12 = src1->nb[2];
    स्थिरांक int64_t nb13 = src1->nb[3];

    स्थिरांक int64_t dnb1   = params->nb1;
    स्थिरांक int64_t dnb2   = params->nb2;
    स्थिरांक int64_t dnb3   = params->nb3;
    स्थिरांक int64_t offset = params->offset;

    स्थिरांक int64_t total_rows = ne11 * ne12 * ne13;

    साठी (int64_t row = thread_id; row < total_rows; row += अंक_threads) {
        स्थिरांक int64_t i1 = row % ne11;
        स्थिरांक int64_t i2 = (row / ne11) % ne12;
        स्थिरांक int64_t i3 = row / (ne11 * ne12);

        स्थिरांक दशांश * src_row = (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src1_data + i1 * nb11 + i2 * nb12 + i3 * nb13);
        दशांश *       dst_row = (दशांश *) ((अक्षर *) dst_data + offset + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        copy_row_aligned(dst_row, src_row, (int32_t) ne10);
    }

    परत 0;
}
