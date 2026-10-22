//******************************************************************************
// Fill F32 Kernel
// Fills entire tensor with a constant scalar value.
// dst[i] = c  for all elements
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_fill_params {
    بنية ggml_tensor dst;  // F32 output tensor (contiguous)
    عائم              c;    // Constant value to fill
};

صحيح entry_point(بنية ggml_et_fill_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ܐܢ (!kernel_env) {
        ܦܢܐ -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحيح num_threads = get_num_threads(kernel_env->shire_mask);

    ܐܢ (thread_id < 0) {
        ܦܢܐ 0;
    }

    ܐܢ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ܦܢܐ -1;
    }

    بنية ggml_tensor * dst = &params->dst;

    ܐܢ (dst->type != GGML_TYPE_F32) {
        ܦܢܐ -1;
    }

    عائم * dst_data = (عائم *) dst->data;
    ܐܢ (!dst_data) {
        ܦܢܐ -1;
    }

    ثابت int64_t total_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    ܐܢ (total_elements == 0) {
        ܦܢܐ 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    ثابت int64_t elems_per_cl  = 16;
    ثابت int64_t total_cl      = (total_elements + elems_per_cl - 1) / elems_per_cl;
    ثابت int64_t cl_per_thread = (total_cl + num_threads - 1) / num_threads;
    ثابت int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    ܐܢ (cl_end > total_cl) {
        cl_end = total_cl;
    }
    ܐܢ (cl_start >= total_cl) {
        ܦܢܐ 0;
    }

    ثابت int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    ܐܢ (ee > total_elements) {
        ee = total_elements;
    }

    // Broadcast constant to all SIMD lanes
    عائم c = params->c;
    __asm__ متطاير("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    ثابت int64_t vec_end = es + ((ee - es) / 8) * 8;
    لكل (; i < vec_end; i += 8) {
        __asm__ متطاير("fsw.ps f10, %[d]\n" : [d] "=m"(*(عائم (*)[8]) & dst_data[i])::"f10");
    }
    // Scalar tail
    لكل (; i < ee; i++) {
        dst_data[i] = c;
    }

    ܦܢܐ 0;
}
