//******************************************************************************
// Fill F32 Kernel
// Fills entire tensor with a constant scalar value.
// dst[i] = c  for all elements
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ಸಂರಚನಾ ggml_et_fill_params {
    ಸಂರಚನಾ ggml_tensor dst;  // F32 output tensor (contiguous)
    ಭಿನ್ನ              c;    // Constant value to fill
};

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_fill_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಯದಿ (!kernel_env) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಪೂರ್ಣಾಙ್ಕ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_threads = get_ಅಂಕೆ_threads(kernel_env->shire_mask);

    ಯದಿ (thread_id < 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಯದಿ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಸಂರಚನಾ ggml_tensor * dst = &params->dst;

    ಯದಿ (dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಭಿನ್ನ * dst_data = (ಭಿನ್ನ *) dst->data;
    ಯದಿ (!dst_data) {
        ಪ್ರತ್ಯಾಗಮ -1;
    }

    ಧ್ರುವ int64_t total_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    ಯದಿ (total_elements == 0) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    ಧ್ರುವ int64_t elems_per_cl  = 16;
    ಧ್ರುವ int64_t total_cl      = (total_elements + elems_per_cl - 1) / elems_per_cl;
    ಧ್ರುವ int64_t cl_per_thread = (total_cl + ಅಂಕೆ_threads - 1) / ಅಂಕೆ_threads;
    ಧ್ರುವ int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    ಯದಿ (cl_end > total_cl) {
        cl_end = total_cl;
    }
    ಯದಿ (cl_start >= total_cl) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಧ್ರುವ int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    ಯದಿ (ee > total_elements) {
        ee = total_elements;
    }

    // Broadcast constant to all SIMD lanes
    ಭಿನ್ನ c = params->c;
    __asm__ ಚಞ್ಚಲ("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    ಧ್ರುವ int64_t vec_end = es + ((ee - es) / 8) * 8;
    ಪ್ರತಿ (; i < vec_end; i += 8) {
        __asm__ ಚಞ್ಚಲ("fsw.ps f10, %[d]\n" : [d] "=m"(*(ಭಿನ್ನ (*)[8]) & dst_data[i])::"f10");
    }
    // Scalar tail
    ಪ್ರತಿ (; i < ee; i++) {
        dst_data[i] = c;
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}
