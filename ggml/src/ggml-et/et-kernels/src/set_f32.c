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

#समावेश "ggml_tensor.h"
#समावेश "platform.h"

#समावेश <stdint.h>

काष्ठा ggml_et_set_params {
    काष्ठा ggml_tensor src1;
    काष्ठा ggml_tensor dst;
    int32_t            nb1;
    int32_t            nb2;
    int32_t            nb3;
    int32_t            offset;
};

अटल अंतरभूत व्योम copy_row_aligned(भग्न * dst, स्थिर भग्न * src, int32_t n) {
    क्रम (int32_t i = 0; i < n; i += 8) {
        __asm__ अस्थिर(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(भग्न (*)[8]) & dst[i])
            : [src_vec] "m"(*(स्थिर भग्न (*)[8]) & src[i])
            : "f11");
    }
}

पूर्णांक entry_point(काष्ठा ggml_et_set_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस 0;
    }

    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;
    }

    काष्ठा ggml_tensor * src1 = &params->src1;
    काष्ठा ggml_tensor * dst  = &params->dst;

    अगर (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस -1;
    }

    स्थिर भग्न * src1_तथ्य = (स्थिर भग्न *) src1->तथ्य;
    भग्न *       dst_तथ्य  = (भग्न *) dst->तथ्य;
    अगर (!src1_तथ्य || !dst_तथ्य) {
        वापस -1;
    }

    स्थिर int64_t ne10 = src1->ne[0];
    स्थिर int64_t ne11 = src1->ne[1];
    स्थिर int64_t ne12 = src1->ne[2];
    स्थिर int64_t ne13 = src1->ne[3];

    अगर (src1->nb[0] != माप(भग्न) || dst->nb[0] != माप(भग्न) || ne10 % 16 != 0) {
        वापस -1;
    }

    स्थिर int64_t nb11 = src1->nb[1];
    स्थिर int64_t nb12 = src1->nb[2];
    स्थिर int64_t nb13 = src1->nb[3];

    स्थिर int64_t dnb1   = params->nb1;
    स्थिर int64_t dnb2   = params->nb2;
    स्थिर int64_t dnb3   = params->nb3;
    स्थिर int64_t offset = params->offset;

    स्थिर int64_t योग_rows = ne11 * ne12 * ne13;

    क्रम (int64_t row = thread_id; row < योग_rows; row += अंक_threads) {
        स्थिर int64_t i1 = row % ne11;
        स्थिर int64_t i2 = (row / ne11) % ne12;
        स्थिर int64_t i3 = row / (ne11 * ne12);

        स्थिर भग्न * src_row = (स्थिर भग्न *) ((स्थिर अक्षर *) src1_तथ्य + i1 * nb11 + i2 * nb12 + i3 * nb13);
        भग्न *       dst_row = (भग्न *) ((अक्षर *) dst_तथ्य + offset + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        copy_row_aligned(dst_row, src_row, (int32_t) ne10);
    }

    वापस 0;
}
