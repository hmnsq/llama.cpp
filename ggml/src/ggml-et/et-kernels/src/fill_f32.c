//******************************************************************************
// Fill F32 Kernel
// Fills entire tensor with a constant scalar value.
// dst[i] = c  for all elements
//******************************************************************************

#समावेश "ggml_tensor.h"
#समावेश "platform.h"

#समावेश <stdint.h>

काष्ठा ggml_et_fill_params {
    काष्ठा ggml_tensor dst;  // F32 output tensor (contiguous)
    भग्न              c;    // Constant value to fill
};

पूर्णांक entry_point(काष्ठा ggml_et_fill_params * params, व्योम * env) {
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

    काष्ठा ggml_tensor * dst = &params->dst;

    अगर (dst->type != GGML_TYPE_F32) {
        वापस -1;
    }

    भग्न * dst_तथ्य = (भग्न *) dst->तथ्य;
    अगर (!dst_तथ्य) {
        वापस -1;
    }

    स्थिर int64_t योग_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    अगर (योग_elements == 0) {
        वापस 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    स्थिर int64_t elems_per_cl  = 16;
    स्थिर int64_t योग_cl      = (योग_elements + elems_per_cl - 1) / elems_per_cl;
    स्थिर int64_t cl_per_thread = (योग_cl + अंक_threads - 1) / अंक_threads;
    स्थिर int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    अगर (cl_end > योग_cl) {
        cl_end = योग_cl;
    }
    अगर (cl_start >= योग_cl) {
        वापस 0;
    }

    स्थिर int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    अगर (ee > योग_elements) {
        ee = योग_elements;
    }

    // Broadcast constant to all SIMD lanes
    भग्न c = params->c;
    __asm__ अस्थिर("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    स्थिर int64_t vec_end = es + ((ee - es) / 8) * 8;
    क्रम (; i < vec_end; i += 8) {
        __asm__ अस्थिर("fsw.ps f10, %[d]\n" : [d] "=m"(*(भग्न (*)[8]) & dst_तथ्य[i])::"f10");
    }
    // Scalar tail
    क्रम (; i < ee; i++) {
        dst_तथ्य[i] = c;
    }

    वापस 0;
}
