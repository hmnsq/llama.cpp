//******************************************************************************
// SUM_ROWS F32 Kernel
// Row-wise sum reduction: dst[0, i1, i2, i3] = sum(src0[0..ne00-1, i1, i2, i3])
// Vectorized 8-wide accumulation with horizontal reduction.
//******************************************************************************

#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <stdint.h>

ഘടന ggml_et_sum_rows_params {
    ഘടന ggml_tensor src0;  // F32 input tensor [ne00, ne01, ne02, ne03]
    ഘടന ggml_tensor dst;   // F32 output tensor [1, ne01, ne02, ne03]
};

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_sum_rows_params * params, ശൂന്യം * env) {
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

    ഘടന ggml_tensor * src0 = &params->src0;
    ഘടന ggml_tensor * dst  = &params->dst;

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        തിരികെ -1;
    }

    ദശാംശം * src0_data = (ദശാംശം *) src0->data;
    ദശാംശം * dst_data  = (ദശാംശം *) dst->data;

    എങ്കിൽ (!src0_data || !dst_data) {
        തിരികെ -1;
    }


    സ്ഥിരം int64_t ne00 = src0->ne[0];  // Row length (to be summed)
    സ്ഥിരം int64_t ne01 = src0->ne[1];
    സ്ഥിരം int64_t ne02 = src0->ne[2];
    സ്ഥിരം int64_t ne03 = src0->ne[3];

    സ്ഥിരം size_t nb01 = src0->nb[1];
    സ്ഥിരം size_t nb02 = src0->nb[2];
    സ്ഥിരം size_t nb03 = src0->nb[3];

    സ്ഥിരം size_t nb1 = dst->nb[1];
    സ്ഥിരം size_t nb2 = dst->nb[2];
    സ്ഥിരം size_t nb3 = dst->nb[3];

    // Flatten rows across dimensions 1,2,3 and distribute across threads
    സ്ഥിരം int64_t total_rows = ne01 * ne02 * ne03;

    വേണ്ടി (int64_t ir = thread_id; ir < total_rows; ir += അക്കം_threads) {
        സ്ഥിരം int64_t i03 = ir / (ne02 * ne01);
        സ്ഥിരം int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        സ്ഥിരം int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

        സ്ഥിരം ദശാംശം * src_row = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
        ദശാംശം *       dst_ptr = (ദശാംശം *) ((അക്ഷരം *) dst_data + i01 * nb1 + i02 * nb2 + i03 * nb3);

        // Vectorized 8-wide sum accumulation
        ദശാംശം zero = 0.0f;
        __asm__ അസ്ഥിര("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        വേണ്ടി (int32_t i0 = 0; i0 < (int32_t) ne00; i0 += 8) {
            __asm__ അസ്ഥിര(
                "flw.ps f11, %[x_vec]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x_vec] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) & src_row[i0])
                : "f10", "f11");
        }

        // Horizontal sum of 8 accumulated values in f10
        ദശാംശം row_sum;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(row_sum)::"t0", "f1", "f2", "f3", "f4", "f5");

        atomic_store_f32(dst_ptr, row_sum);
    }

    തിരികെ 0;
}
