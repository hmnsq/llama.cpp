//******************************************************************************
// SQR F32 Kernel
// Element-wise square: y[i] = x[i] * x[i]
//******************************************************************************

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>

// SQR kernel parameters structure (unary op: src0 -> dst)
ସଂରଚନା ggml_et_sqr_params {
    ସଂରଚନା ggml_tensor src0;  // F32 input tensor
    ସଂରଚନା ggml_tensor dst;   // F32 output tensor
};

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_sqr_params * params, ଶୂନ୍ୟ * env) {
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
        ଫେରନ୍ତୁ -1;  // Invalid pointer
    }

    ସଂରଚନା ggml_tensor * src0 = &params->src0;
    ସଂରଚନା ggml_tensor * dst  = &params->dst;

    ଯଦି (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ -1;  // Unsupported type combination
    }

    ଦଶମିକ * src0_data = (ଦଶମିକ *) src0->data;
    ଦଶମିକ * dst_data  = (ଦଶମିକ *) dst->data;

    ଯଦି (!src0_data || !dst_data) {
        ଫେରନ୍ତୁ -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    ଅଚଳ int64_t total_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ଅଚଳ int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes per float
    ଅଚଳ int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    ଅଚଳ int64_t cl_per_thread = (total_cachelines + ଅଙ୍କ_threads - 1) / ଅଙ୍କ_threads;
    ଅଚଳ int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    ଯଦି (cl_end > total_cachelines) {
        cl_end = total_cachelines;
    }

    ଯଦି (cl_start >= total_cachelines) {
        ଫେରନ୍ତୁ 0;
    }

    ଅଚଳ int64_t elem_start = cl_start * elements_per_cacheline;
    int64_t       elem_end   = cl_end * elements_per_cacheline;
    ଯଦି (elem_end > total_elements) {
        elem_end = total_elements;
    }

    ଅଚଳ ଦଶମିକ * src_ptr = src0_data + elem_start;
    ଦଶମିକ *       dst_ptr = dst_data + elem_start;
    ଅଚଳ int32_t ଗଣନା_2   = (int32_t) (elem_end - elem_start);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    ପାଇଁ (int32_t i0 = 0; i0 < ଗଣନା_2; i0 += 8) {
        __asm__ ଅସ୍ଥିର(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [result] "=m"(*(ଦଶମିକ (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    ଫେରନ୍ତୁ 0;
}
