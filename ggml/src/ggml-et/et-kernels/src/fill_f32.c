//******************************************************************************
// Fill F32 Kernel
// Fills entire tensor with a constant scalar value.
// dst[i] = c  for all elements
//******************************************************************************

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>

ସଂରଚନା ggml_et_fill_params {
    ସଂରଚନା ggml_tensor dst;  // F32 output tensor (contiguous)
    ଦଶମିକ              c;    // Constant value to fill
};

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_fill_params * params, ଶୂନ୍ୟ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ଯଦି (!kernel_env) {
        ଫେରନ୍ତୁ -1;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_threads = get_ଅଙ୍କ_threads(kernel_env->shire_mask);

    ଯଦି (thread_id < 0) {
        ଫେରନ୍ତୁ 0;
    }

    ଯଦି (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ଫେରନ୍ତୁ -1;
    }

    ସଂରଚନା ggml_tensor * dst = &params->dst;

    ଯଦି (dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ -1;
    }

    ଦଶମିକ * dst_data = (ଦଶମିକ *) dst->data;
    ଯଦି (!dst_data) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int64_t total_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    ଯଦି (total_elements == 0) {
        ଫେରନ୍ତୁ 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    ଅଚଳ int64_t elems_per_cl  = 16;
    ଅଚଳ int64_t total_cl      = (total_elements + elems_per_cl - 1) / elems_per_cl;
    ଅଚଳ int64_t cl_per_thread = (total_cl + ଅଙ୍କ_threads - 1) / ଅଙ୍କ_threads;
    ଅଚଳ int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    ଯଦି (cl_end > total_cl) {
        cl_end = total_cl;
    }
    ଯଦି (cl_start >= total_cl) {
        ଫେରନ୍ତୁ 0;
    }

    ଅଚଳ int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    ଯଦି (ee > total_elements) {
        ee = total_elements;
    }

    // Broadcast constant to all SIMD lanes
    ଦଶମିକ c = params->c;
    __asm__ ଅସ୍ଥିର("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    ଅଚଳ int64_t vec_end = es + ((ee - es) / 8) * 8;
    ପାଇଁ (; i < vec_end; i += 8) {
        __asm__ ଅସ୍ଥିର("fsw.ps f10, %[d]\n" : [d] "=m"(*(ଦଶମିକ (*)[8]) & dst_data[i])::"f10");
    }
    // Scalar tail
    ପାଇଁ (; i < ee; i++) {
        dst_data[i] = c;
    }

    ଫେରନ୍ତୁ 0;
}
