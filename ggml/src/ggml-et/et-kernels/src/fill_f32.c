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

पूर्णाङ्क entry_point(संरचना ggml_et_fill_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        फर्काऊ -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अङ्क_threads = get_अङ्क_threads(kernel_env->shire_mask);

    यदि (thread_id < 0) {
        फर्काऊ 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        फर्काऊ -1;
    }

    संरचना ggml_tensor * dst = &params->dst;

    यदि (dst->type != GGML_TYPE_F32) {
        फर्काऊ -1;
    }

    भिन्न * dst_data = (भिन्न *) dst->data;
    यदि (!dst_data) {
        फर्काऊ -1;
    }

    अचर int64_t total_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    यदि (total_elements == 0) {
        फर्काऊ 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    अचर int64_t elems_per_cl  = 16;
    अचर int64_t total_cl      = (total_elements + elems_per_cl - 1) / elems_per_cl;
    अचर int64_t cl_per_thread = (total_cl + अङ्क_threads - 1) / अङ्क_threads;
    अचर int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    यदि (cl_end > total_cl) {
        cl_end = total_cl;
    }
    यदि (cl_start >= total_cl) {
        फर्काऊ 0;
    }

    अचर int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    यदि (ee > total_elements) {
        ee = total_elements;
    }

    // Broadcast constant to all SIMD lanes
    भिन्न c = params->c;
    __asm__ चञ्चल("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    अचर int64_t vec_end = es + ((ee - es) / 8) * 8;
    लागि (; i < vec_end; i += 8) {
        __asm__ चञ्चल("fsw.ps f10, %[d]\n" : [d] "=m"(*(भिन्न (*)[8]) & dst_data[i])::"f10");
    }
    // Scalar tail
    लागि (; i < ee; i++) {
        dst_data[i] = c;
    }

    फर्काऊ 0;
}
