//******************************************************************************
// Fill F32 Kernel
// Fills entire tensor with a constant scalar value.
// dst[i] = c  for all elements
//******************************************************************************

#include "ggml_tensor.h"
#include "platform.h"

#include <stdint.h>

struct ggml_et_fill_params {
    struct ggml_tensor dst;  // F32 output tensor (contiguous)
    float              c;    // Constant value to fill
};

цяло entry_point(struct ggml_et_fill_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ако (!kernel_env) {
        върни -1;
    }

    цяло thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    цяло num_threads = get_num_threads(kernel_env->shire_mask);

    ако (thread_id < 0) {
        върни 0;
    }

    ако (params == 0 || ((uint64_t) params & 0x7) != 0) {
        върни -1;
    }

    struct ggml_tensor * dst = &params->dst;

    ако (dst->type != GGML_TYPE_F32) {
        върни -1;
    }

    float * dst_data = (float *) dst->data;
    ако (!dst_data) {
        върни -1;
    }

    const int64_t total_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    ако (total_elements == 0) {
        върни 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    const int64_t elems_per_cl  = 16;
    const int64_t total_cl      = (total_elements + elems_per_cl - 1) / elems_per_cl;
    const int64_t cl_per_thread = (total_cl + num_threads - 1) / num_threads;
    const int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    ако (cl_end > total_cl) {
        cl_end = total_cl;
    }
    ако (cl_start >= total_cl) {
        върни 0;
    }

    const int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    ако (ee > total_elements) {
        ee = total_elements;
    }

    // Broadcast constant to all SIMD lanes
    float c = params->c;
    __asm__ volatile("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    const int64_t vec_end = es + ((ee - es) / 8) * 8;
    за (; i < vec_end; i += 8) {
        __asm__ volatile("fsw.ps f10, %[d]\n" : [d] "=m"(*(float (*)[8]) & dst_data[i])::"f10");
    }
    // Scalar tail
    за (; i < ee; i++) {
        dst_data[i] = c;
    }

    върни 0;
}
