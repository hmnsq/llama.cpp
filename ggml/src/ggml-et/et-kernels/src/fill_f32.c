//******************************************************************************
// Fill F32 Kernel
// Fills entire tensor with a constant scalar value.
// dst[i] = c  for all elements
//******************************************************************************

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <stdint.h>

સંરચના ggml_et_fill_params {
    સંરચના ggml_tensor dst;  // F32 output tensor (contiguous)
    દશાંશ              c;    // Constant value to fill
};

પૂર્ણાંક entry_point(સંરચના ggml_et_fill_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    જો (!kernel_env) {
        પાછા -1;
    }

    પૂર્ણાંક thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    પૂર્ણાંક અંક_threads = get_અંક_threads(kernel_env->shire_mask);

    જો (thread_id < 0) {
        પાછા 0;
    }

    જો (params == 0 || ((uint64_t) params & 0x7) != 0) {
        પાછા -1;
    }

    સંરચના ggml_tensor * dst = &params->dst;

    જો (dst->type != GGML_TYPE_F32) {
        પાછા -1;
    }

    દશાંશ * dst_data = (દશાંશ *) dst->data;
    જો (!dst_data) {
        પાછા -1;
    }

    અચલ int64_t total_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    જો (total_elements == 0) {
        પાછા 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    અચલ int64_t elems_per_cl  = 16;
    અચલ int64_t total_cl      = (total_elements + elems_per_cl - 1) / elems_per_cl;
    અચલ int64_t cl_per_thread = (total_cl + અંક_threads - 1) / અંક_threads;
    અચલ int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    જો (cl_end > total_cl) {
        cl_end = total_cl;
    }
    જો (cl_start >= total_cl) {
        પાછા 0;
    }

    અચલ int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    જો (ee > total_elements) {
        ee = total_elements;
    }

    // Broadcast constant to all SIMD lanes
    દશાંશ c = params->c;
    __asm__ અસ્થિર("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    અચલ int64_t vec_end = es + ((ee - es) / 8) * 8;
    માટે (; i < vec_end; i += 8) {
        __asm__ અસ્થિર("fsw.ps f10, %[d]\n" : [d] "=m"(*(દશાંશ (*)[8]) & dst_data[i])::"f10");
    }
    // Scalar tail
    માટે (; i < ee; i++) {
        dst_data[i] = c;
    }

    પાછા 0;
}
