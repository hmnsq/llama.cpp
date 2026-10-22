//******************************************************************************
// SQR F32 Kernel
// Element-wise square: y[i] = x[i] * x[i]
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

// SQR kernel parameters structure (unary op: src0 -> dst)
ಸಂರಚನಾ ggml_et_sqr_params {
    ಸಂರಚನಾ ggml_tensor src0;  // F32 input tensor
    ಸಂರಚನಾ ggml_tensor dst;   // F32 output tensor
};

ಪೂರ್ಣಾಙ್ಕ entry_point(ಸಂರಚನಾ ggml_et_sqr_params * params, ಶೂನ್ಯ * env) {
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
        ಪ್ರತ್ಯಾಗಮ -1;  // Invalid pointer
    }

    ಸಂರಚನಾ ggml_tensor * src0 = &params->src0;
    ಸಂರಚನಾ ggml_tensor * dst  = &params->dst;

    ಯದಿ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Unsupported type combination
    }

    ಭಿನ್ನ * src0_data = (ಭಿನ್ನ *) src0->data;
    ಭಿನ್ನ * dst_data  = (ಭಿನ್ನ *) dst->data;

    ಯದಿ (!src0_data || !dst_data) {
        ಪ್ರತ್ಯಾಗಮ -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    ಧ್ರುವ int64_t total_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ಧ್ರುವ int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes per float
    ಧ್ರುವ int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    ಧ್ರುವ int64_t cl_per_thread = (total_cachelines + ಅಂಕೆ_threads - 1) / ಅಂಕೆ_threads;
    ಧ್ರುವ int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    ಯದಿ (cl_end > total_cachelines) {
        cl_end = total_cachelines;
    }

    ಯದಿ (cl_start >= total_cachelines) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಧ್ರುವ int64_t elem_start = cl_start * elements_per_cacheline;
    int64_t       elem_end   = cl_end * elements_per_cacheline;
    ಯದಿ (elem_end > total_elements) {
        elem_end = total_elements;
    }

    ಧ್ರುವ ಭಿನ್ನ * src_ptr = src0_data + elem_start;
    ಭಿನ್ನ *       dst_ptr = dst_data + elem_start;
    ಧ್ರುವ int32_t ಎಣಿಕೆ   = (int32_t) (elem_end - elem_start);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    ಪ್ರತಿ (int32_t i0 = 0; i0 < ಎಣಿಕೆ; i0 += 8) {
        __asm__ ಚಞ್ಚಲ(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [result] "=m"(*(ಭಿನ್ನ (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(ಧ್ರುವ ಭಿನ್ನ (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    ಪ್ರತ್ಯಾಗಮ 0;
}
