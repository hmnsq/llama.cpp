//******************************************************************************
// CLAMP F32 Kernel
// Element-wise: dst[i] = min(max(src0[i], min_val), max_val)
//******************************************************************************

#include "ggml_tensor.h"
#include "platform.h"

#include <stdint.h>

sa_nracanaa ggml_et_clamp_params {
    sa_nracanaa ggml_tensor src0;  // F32 input  (contiguous)
    sa_nracanaa ggml_tensor dst;   // F32 output (contiguous; may alias src0.data)
    float              min_val;
    float              max_val;
};

// Vectorized fmax/fmin clamp with scalar tail. n may be any non-negative int.
static anta_hstha void clamp_block_f32(float * dst, const float * src, float min_val, float max_val, int32_t n) {
    int32_t       i       = 0;
    const int32_t vec_end = (n / 8) * 8;

    ប្រសិនបើ (vec_end > 0) {
        unsigned long temp_mask;
        __asm__ ca_ncala("mova.x.m %0" : "=r"(temp_mask));
        __asm__ ca_ncala("mov.m.x m0, x0, 0xFF");

        សម្រាប់ (; i < vec_end; i += 8) {
            __asm__ ca_ncala(
                "flw.ps  f10, %[s]\n"
                "fbc.ps  f11, %[mn]\n"
                "fbc.ps  f12, %[mx]\n"
                "fmax.ps f13, f10, f11\n"
                "fmin.ps f13, f13, f12\n"
                "fsw.ps  f13, %[d]\n"
                : [d] "=m"(*(float (*)[8]) & dst[i])
                : [s] "m"(*(const float (*)[8]) & src[i]), [mn] "m"(min_val), [mx] "m"(max_val)
                : "f10", "f11", "f12", "f13");
        }

        __asm__ ca_ncala("mova.m.x %0" ::"r"(temp_mask));
    }

    សម្រាប់ (; i < n; i++) {
        float v = src[i];
        ប្រសិនបើ (v < min_val) {
            v = min_val;
        }
        ប្រសិនបើ (v > max_val) {
            v = max_val;
        }
        dst[i] = v;
    }
}

ចំនួនគត់ entry_point(sa_nracanaa ggml_et_clamp_params * params, void * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ប្រសិនបើ (!kernel_env) {
        ត្រឡប់ -1;
    }

    ចំនួនគត់ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ចំនួនគត់ num_threads = get_num_threads(kernel_env->shire_mask);

    ប្រសិនបើ (thread_id < 0) {
        ត្រឡប់ 0;
    }

    ប្រសិនបើ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ត្រឡប់ -1;
    }

    sa_nracanaa ggml_tensor * src0 = &params->src0;
    sa_nracanaa ggml_tensor * dst  = &params->dst;

    ប្រសិនបើ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ត្រឡប់ -1;
    }

    float * src0_data = (float *) src0->data;
    float * dst_data  = (float *) dst->data;

    ប្រសិនបើ (!src0_data || !dst_data) {
        ត្រឡប់ -1;
    }

    const int64_t total_elements = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    ប្រសិនបើ (total_elements <= 0) {
        ត្រឡប់ 0;
    }

    const float min_val = params->min_val;
    const float max_val = params->max_val;

    // Distribute by cache lines (16 F32 elements). Each thread owns disjoint
    // cache lines, so a partial trailing line is written by exactly one
    // thread — safe under non-coherent caches.
    const int64_t elems_per_cl = 16;
    const int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

    const int64_t cl_per_thread = (total_cl + num_threads - 1) / num_threads;
    const int64_t cl_start      = (int64_t) thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    ប្រសិនបើ (cl_end > total_cl) {
        cl_end = total_cl;
    }
    ប្រសិនបើ (cl_start >= total_cl) {
        ត្រឡប់ 0;
    }

    const int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    ប្រសិនបើ (ee > total_elements) {
        ee = total_elements;
    }

    clamp_block_f32(dst_data + es, src0_data + es, min_val, max_val, (int32_t) (ee - es));
    ត្រឡប់ 0;
}
