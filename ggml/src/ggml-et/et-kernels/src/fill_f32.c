//******************************************************************************
// Fill F32 Kernel
// Fills entire tensor with a constant scalar value.
// dst[i] = c  for all elements
//******************************************************************************

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <stdint.h>

கட்டமைப்பு ggml_et_fill_params {
    கட்டமைப்பு ggml_tensor dst;  // F32 output tensor (contiguous)
    தசம              c;    // Constant value to fill
};

முழு entry_point(கட்டமைப்பு ggml_et_fill_params * params, வெறுமை * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    எனில் (!kernel_env) {
        திரும்பு -1;
    }

    முழு thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    முழு இலக்கம்_threads = get_இலக்கம்_threads(kernel_env->shire_mask);

    எனில் (thread_id < 0) {
        திரும்பு 0;
    }

    எனில் (params == 0 || ((uint64_t) params & 0x7) != 0) {
        திரும்பு -1;
    }

    கட்டமைப்பு ggml_tensor * dst = &params->dst;

    எனில் (dst->type != GGML_TYPE_F32) {
        திரும்பு -1;
    }

    தசம * dst_data = (தசம *) dst->data;
    எனில் (!dst_data) {
        திரும்பு -1;
    }

    மாறிலி int64_t total_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    எனில் (total_elements == 0) {
        திரும்பு 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    மாறிலி int64_t elems_per_cl  = 16;
    மாறிலி int64_t total_cl      = (total_elements + elems_per_cl - 1) / elems_per_cl;
    மாறிலி int64_t cl_per_thread = (total_cl + இலக்கம்_threads - 1) / இலக்கம்_threads;
    மாறிலி int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    எனில் (cl_end > total_cl) {
        cl_end = total_cl;
    }
    எனில் (cl_start >= total_cl) {
        திரும்பு 0;
    }

    மாறிலி int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    எனில் (ee > total_elements) {
        ee = total_elements;
    }

    // Broadcast constant to all SIMD lanes
    தசம c = params->c;
    __asm__ நிலையற்ற("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    மாறிலி int64_t vec_end = es + ((ee - es) / 8) * 8;
    க்கு (; i < vec_end; i += 8) {
        __asm__ நிலையற்ற("fsw.ps f10, %[d]\n" : [d] "=m"(*(தசம (*)[8]) & dst_data[i])::"f10");
    }
    // Scalar tail
    க்கு (; i < ee; i++) {
        dst_data[i] = c;
    }

    திரும்பு 0;
}
