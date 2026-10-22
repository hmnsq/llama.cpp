//******************************************************************************
// SET F32 Kernel
// Minimal ET implementation for inplace F32 SET into a contiguous destination
// using a contiguous F32 source view and explicit destination view strides.
//
// Supported shape family:
// - dst/base is contiguous F32
// - src1 is contiguous F32
// - src1.ne[0] is cacheline-aligned (multiple of 16 floats)
// - destination view strides/offset are cacheline-aligned
//******************************************************************************

#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <stdint.h>

ഘടന ggml_et_set_params {
    ഘടന ggml_tensor src1;
    ഘടന ggml_tensor dst;
    int32_t            nb1;
    int32_t            nb2;
    int32_t            nb3;
    int32_t            offset;
};

സ്ഥിതം വരിയിൽ ശൂന്യം copy_row_aligned(ദശാംശം * dst, സ്ഥിരം ദശാംശം * src, int32_t n) {
    വേണ്ടി (int32_t i = 0; i < n; i += 8) {
        __asm__ അസ്ഥിര(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(ദശാംശം (*)[8]) & dst[i])
            : [src_vec] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) & src[i])
            : "f11");
    }
}

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_set_params * params, ശൂന്യം * env) {
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

    ഘടന ggml_tensor * src1 = &params->src1;
    ഘടന ggml_tensor * dst  = &params->dst;

    എങ്കിൽ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        തിരികെ -1;
    }

    സ്ഥിരം ദശാംശം * src1_data = (സ്ഥിരം ദശാംശം *) src1->data;
    ദശാംശം *       dst_data  = (ദശാംശം *) dst->data;
    എങ്കിൽ (!src1_data || !dst_data) {
        തിരികെ -1;
    }

    സ്ഥിരം int64_t ne10 = src1->ne[0];
    സ്ഥിരം int64_t ne11 = src1->ne[1];
    സ്ഥിരം int64_t ne12 = src1->ne[2];
    സ്ഥിരം int64_t ne13 = src1->ne[3];

    എങ്കിൽ (src1->nb[0] != വലുപ്പം(ദശാംശം) || dst->nb[0] != വലുപ്പം(ദശാംശം) || ne10 % 16 != 0) {
        തിരികെ -1;
    }

    സ്ഥിരം int64_t nb11 = src1->nb[1];
    സ്ഥിരം int64_t nb12 = src1->nb[2];
    സ്ഥിരം int64_t nb13 = src1->nb[3];

    സ്ഥിരം int64_t dnb1   = params->nb1;
    സ്ഥിരം int64_t dnb2   = params->nb2;
    സ്ഥിരം int64_t dnb3   = params->nb3;
    സ്ഥിരം int64_t offset = params->offset;

    സ്ഥിരം int64_t total_rows = ne11 * ne12 * ne13;

    വേണ്ടി (int64_t row = thread_id; row < total_rows; row += അക്കം_threads) {
        സ്ഥിരം int64_t i1 = row % ne11;
        സ്ഥിരം int64_t i2 = (row / ne11) % ne12;
        സ്ഥിരം int64_t i3 = row / (ne11 * ne12);

        സ്ഥിരം ദശാംശം * src_row = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src1_data + i1 * nb11 + i2 * nb12 + i3 * nb13);
        ദശാംശം *       dst_row = (ദശാംശം *) ((അക്ഷരം *) dst_data + offset + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        copy_row_aligned(dst_row, src_row, (int32_t) ne10);
    }

    തിരികെ 0;
}
