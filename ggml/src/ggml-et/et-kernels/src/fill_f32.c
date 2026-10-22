//******************************************************************************
// Fill F32 Kernel
// Fills entire tensor with a constant scalar value.
// dst[i] = c  for all elements
//******************************************************************************

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdint.h>

संरचना ggml_et_fill_params {
    संरचना ggml_tensor dst;  // F32 output tensor (contiguous)
    भिन्न              c;    // Constant value to fill
};

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_fill_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_आवरणम्);
    पूर्णाङ्क अङ्कः_तन्तवः = get_अङ्कः_तन्तवः(kernel_env->shire_आवरणम्);

    यदि (thread_id < 0) {
        प्रत्यागम 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;
    }

    संरचना ggml_tensor * dst = &params->dst;

    यदि (dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    भिन्न * dst_दत्त = (भिन्न *) dst->दत्त;
    यदि (!dst_दत्त) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t योग_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    यदि (योग_elements == 0) {
        प्रत्यागम 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    ध्रुव int64_t elems_प्रत्येकम्_cl  = 16;
    ध्रुव int64_t योग_cl      = (योग_elements + elems_प्रत्येकम्_cl - 1) / elems_प्रत्येकम्_cl;
    ध्रुव int64_t cl_प्रत्येकम्_thread = (योग_cl + अङ्कः_तन्तवः - 1) / अङ्कः_तन्तवः;
    ध्रुव int64_t cl_start      = thread_id * cl_प्रत्येकम्_thread;
    int64_t       cl_अन्तः        = cl_start + cl_प्रत्येकम्_thread;
    यदि (cl_अन्तः > योग_cl) {
        cl_अन्तः = योग_cl;
    }
    यदि (cl_start >= योग_cl) {
        प्रत्यागम 0;
    }

    ध्रुव int64_t es = cl_start * elems_प्रत्येकम्_cl;
    int64_t       ee = cl_अन्तः * elems_प्रत्येकम्_cl;
    यदि (ee > योग_elements) {
        ee = योग_elements;
    }

    // Broadcast constant to all SIMD lanes
    भिन्न c = params->c;
    __asm__ चञ्चल("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    ध्रुव int64_t vec_अन्तः = es + ((ee - es) / 8) * 8;
    प्रति (; i < vec_अन्तः; i += 8) {
        __asm__ चञ्चल("fsw.ps f10, %[d]\n" : [d] "=m"(*(भिन्न (*)[8]) & dst_दत्त[i])::"f10");
    }
    // Scalar tail
    प्रति (; i < ee; i++) {
        dst_दत्त[i] = c;
    }

    प्रत्यागम 0;
}
