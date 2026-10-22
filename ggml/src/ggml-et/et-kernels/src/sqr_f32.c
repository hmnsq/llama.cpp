//******************************************************************************
// SQR F32 Kernel
// Element-wise square: y[i] = x[i] * x[i]
//******************************************************************************

#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <stdint.h>

// SQR kernel parameters structure (unary op: src0 -> dst)
ഘടന ggml_et_sqr_params {
    ഘടന ggml_tensor src0;  // F32 input tensor
    ഘടന ggml_tensor dst;   // F32 output tensor
};

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_sqr_params * params, ശൂന്യം * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    എങ്കിൽ (!kernel_env) {
        തിരികെ -1;
    }

    പൂർണ്ണസംഖ്യ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    പൂർണ്ണസംഖ്യ അക്കം_threads = get_അക്കം_threads(kernel_env->shire_mask);

    എങ്കിൽ (thread_id < 0) {
        തിരികെ 0;
    }

    എങ്കിൽ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        തിരികെ -1;  // Invalid pointer
    }

    ഘടന ggml_tensor * src0 = &params->src0;
    ഘടന ggml_tensor * dst  = &params->dst;

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        തിരികെ -1;  // Unsupported type combination
    }

    ദശാംശം * src0_data = (ദശാംശം *) src0->data;
    ദശാംശം * dst_data  = (ദശാംശം *) dst->data;

    എങ്കിൽ (!src0_data || !dst_data) {
        തിരികെ -1;  // Null data pointer
    }

    // Both src and dst are contiguous F32: flatten and distribute by cache lines
    സ്ഥിരം int64_t total_elements         = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    സ്ഥിരം int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes per float
    സ്ഥിരം int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    സ്ഥിരം int64_t cl_per_thread = (total_cachelines + അക്കം_threads - 1) / അക്കം_threads;
    സ്ഥിരം int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    എങ്കിൽ (cl_end > total_cachelines) {
        cl_end = total_cachelines;
    }

    എങ്കിൽ (cl_start >= total_cachelines) {
        തിരികെ 0;
    }

    സ്ഥിരം int64_t elem_start = cl_start * elements_per_cacheline;
    int64_t       elem_end   = cl_end * elements_per_cacheline;
    എങ്കിൽ (elem_end > total_elements) {
        elem_end = total_elements;
    }

    സ്ഥിരം ദശാംശം * src_ptr = src0_data + elem_start;
    ദശാംശം *       dst_ptr = dst_data + elem_start;
    സ്ഥിരം int32_t എണ്ണം   = (int32_t) (elem_end - elem_start);

    // Process 8 elements at a time: dst[i] = src[i] * src[i]
    വേണ്ടി (int32_t i0 = 0; i0 < എണ്ണം; i0 += 8) {
        __asm__ അസ്ഥിര(
            "flw.ps f10, %[x_vec]\n"   // Load 8 input values
            "fmul.ps f11, f10, f10\n"  // x * x (8-wide)
            "fsw.ps f11, %[result]\n"  // Store 8 results

            : [result] "=m"(*(ദശാംശം (*)[8]) & dst_ptr[i0])
            : [x_vec] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) & src_ptr[i0])
            : "f10", "f11");
    }

    തിരികെ 0;
}
