//******************************************************************************
// Fill F32 Kernel
// Fills entire tensor with a constant scalar value.
// dst[i] = c  for all elements
//******************************************************************************

#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <stdint.h>

ഘടന ggml_et_fill_params {
    ഘടന ggml_tensor dst;  // F32 output tensor (contiguous)
    ദശാംശം              c;    // Constant value to fill
};

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_fill_params * params, ശൂന്യം * env) {
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
        തിരികെ -1;
    }

    ഘടന ggml_tensor * dst = &params->dst;

    എങ്കിൽ (dst->type != GGML_TYPE_F32) {
        തിരികെ -1;
    }

    ദശാംശം * dst_data = (ദശാംശം *) dst->data;
    എങ്കിൽ (!dst_data) {
        തിരികെ -1;
    }

    സ്ഥിരം int64_t total_elements = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];

    എങ്കിൽ (total_elements == 0) {
        തിരികെ 0;
    }

    // Distribute by cache lines (16 floats = 64 bytes)
    സ്ഥിരം int64_t elems_per_cl  = 16;
    സ്ഥിരം int64_t total_cl      = (total_elements + elems_per_cl - 1) / elems_per_cl;
    സ്ഥിരം int64_t cl_per_thread = (total_cl + അക്കം_threads - 1) / അക്കം_threads;
    സ്ഥിരം int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    എങ്കിൽ (cl_end > total_cl) {
        cl_end = total_cl;
    }
    എങ്കിൽ (cl_start >= total_cl) {
        തിരികെ 0;
    }

    സ്ഥിരം int64_t es = cl_start * elems_per_cl;
    int64_t       ee = cl_end * elems_per_cl;
    എങ്കിൽ (ee > total_elements) {
        ee = total_elements;
    }

    // Broadcast constant to all SIMD lanes
    ദശാംശം c = params->c;
    __asm__ അസ്ഥിര("fbc.ps f10, %[v]\n" : : [v] "m"(c) : "f10");

    // Vector fill (8-wide)
    int64_t       i       = es;
    സ്ഥിരം int64_t vec_end = es + ((ee - es) / 8) * 8;
    വേണ്ടി (; i < vec_end; i += 8) {
        __asm__ അസ്ഥിര("fsw.ps f10, %[d]\n" : [d] "=m"(*(ദശാംശം (*)[8]) & dst_data[i])::"f10");
    }
    // Scalar tail
    വേണ്ടി (; i < ee; i++) {
        dst_data[i] = c;
    }

    തിരികെ 0;
}
