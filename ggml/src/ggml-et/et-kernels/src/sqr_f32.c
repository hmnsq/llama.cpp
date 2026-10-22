//******************************************************************************
// SQR F32 Kernel
// Element-wise square: y[i] = x[i] * x[i]
//******************************************************************************

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <stdint.h>

// SQR kernel parameters structure (unary op: src0 -> dst)
கட்டமைப்பு ggml_et_sqr_params {
    கட்டமைப்பு ggml_tensor src0;  // F32 input tensor
    கட்டமைப்பு ggml_tensor dst;   // F32 output tensor
};

முழு entry_point(கட்டமைப்பு ggml_et_sqr_params * params, வெறுமை * env) {
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
        திரும்பு -1;  // Invalid pointer
    }

    கட்டமைப்பு ggml_tensor * src0 = &params->src0;
    கட்டமைப்பு ggml_tensor * dst  = &params->dst;

    எனில் (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        திரும்பு -1;  // Unsupported type combination
    }

    தசம * src0_data = (தசம *) src0->data;
    தசம * dst_data  = (தசம *) dst->data;

    எனில் (!src0_data || !dst_data) {
        திரும்பு -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    மாறிலி int64_t total_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    மாறிலி int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes per float
    மாறிலி int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    மாறிலி int64_t cl_per_thread = (total_cachelines + இலக்கம்_threads - 1) / இலக்கம்_threads;
    மாறிலி int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    எனில் (cl_end > total_cachelines) {
        cl_end = total_cachelines;
    }

    எனில் (cl_start >= total_cachelines) {
        திரும்பு 0;
    }

    மாறிலி int64_t elem_start = cl_start * elements_per_cacheline;
    int64_t       elem_end   = cl_end * elements_per_cacheline;
    எனில் (elem_end > total_elements) {
        elem_end = total_elements;
    }

    மாறிலி தசம * src_ptr = src0_data + elem_start;
    தசம *       dst_ptr = dst_data + elem_start;
    மாறிலி int32_t மொத்தம்   = (int32_t) (elem_end - elem_start);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    க்கு (int32_t i0 = 0; i0 < மொத்தம்; i0 += 8) {
        __asm__ நிலையற்ற(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [result] "=m"(*(தசம (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(மாறிலி தசம (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    திரும்பு 0;
}
