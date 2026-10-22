//******************************************************************************
// Fill F32 Kernel
// Fills entire tensor with a constant scalar value.
// dst[i] = c  for all elements
//******************************************************************************

#include "ggml_tensor.h"
#include "platform.h"

#include <stdint.h>

sa_nracanaa ggml_et_fill_params {
    sa_nracanaa ggml_tensor dst;  // F32 output tensor (contiguous)
    float              c;    // Constant value to fill
};

ကိန်းပြည့် entry_point(sa_nracanaa ggml_et_fill_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    အကယ်၍ (!kernel_env) {
        ပြန်ပေး -1;
    }

    ကိန်းပြည့် thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ကိန်းပြည့် num_threads = get_num_threads(kernel_env->shire_mask);

    အကယ်၍ (thread_id < 0) {
        ပြန်ပေး 0;
    }

    အကယ်၍ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ပြန်ပေး -1;
    }

    sa_nracanaa ggml_tensor * dst = &params->dst;

    အကယ်၍ (dst->type != GGML_TYPE_F32) {
        ပြန်ပေး -1;
    }

    float * dst_data = (float *) dst->data;
    အကယ်၍ (!dst_data) {
        ပြန်ပေး -1;
    }

    const int64_t total_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    အကယ်၍ (total_elements == 0) {
        ပြန်ပေး 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    const int64_t elems_per_cl  = 16;
    const int64_t total_cl      = (total_elements + elems_per_cl - 1) / elems_per_cl;
    const int64_t cl_per_thread = (total_cl + num_threads - 1) / num_threads;
    const int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    အကယ်၍ (cl_end > total_cl) {
        cl_end = total_cl;
    }
    အကယ်၍ (cl_start >= total_cl) {
        ပြန်ပေး 0;
    }

    const int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    အကယ်၍ (ee > total_elements) {
        ee = total_elements;
    }

    // Broadcast constant to all SIMD lanes
    float c = params->c;
    __asm__ ca_ncala("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    const int64_t vec_end = es + ((ee - es) / 8) * 8;
    အတွက် (; i < vec_end; i += 8) {
        __asm__ ca_ncala("fsw.ps f10, %[d]\n" : [d] "=m"(*(float (*)[8]) & dst_data[i])::"f10");
    }
    // Scalar tail
    အတွက် (; i < ee; i++) {
        dst_data[i] = c;
    }

    ပြန်ပေး 0;
}
