//******************************************************************************
// CUMSUM F32 Kernel
// Computes an inclusive prefix sum along dim 0 for each row in higher dims.
// First-pass implementation: scalar and row-contiguous input/output only.
//******************************************************************************

#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <stdint.h>

ഘടന ggml_et_cumsum_params {
    ഘടന ggml_tensor src0;
    ഘടന ggml_tensor dst;
};

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_cumsum_params * params, ശൂന്യം * env) {
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

    സ്ഥിരം int64_t ne0 = src0->ne[0];
    സ്ഥിരം int64_t ne1 = src0->ne[1];
    സ്ഥിരം int64_t ne2 = src0->ne[2];
    സ്ഥിരം int64_t ne3 = src0->ne[3];

    സ്ഥിരം size_t snb0 = src0->nb[0];
    സ്ഥിരം size_t snb1 = src0->nb[1];
    സ്ഥിരം size_t snb2 = src0->nb[2];
    സ്ഥിരം size_t snb3 = src0->nb[3];

    സ്ഥിരം size_t dnb0 = dst->nb[0];
    സ്ഥിരം size_t dnb1 = dst->nb[1];
    സ്ഥിരം size_t dnb2 = dst->nb[2];
    സ്ഥിരം size_t dnb3 = dst->nb[3];

    എങ്കിൽ (snb0 != വലുപ്പം(ദശാംശം) || dnb0 != വലുപ്പം(ദശാംശം)) {
        തിരികെ -1;
    }

    സ്ഥിരം int64_t total_rows     = ne1 * ne2 * ne3;
    സ്ഥിരം int64_t rows_per_group = et_rows_per_cacheline_group(ne0, വലുപ്പം(ദശാംശം));
    സ്ഥിരം int64_t total_groups   = (total_rows + rows_per_group - 1) / rows_per_group;

    വേണ്ടി (int64_t grp = thread_id; grp < total_groups; grp += അക്കം_threads) {
        സ്ഥിരം int64_t row_start = grp * rows_per_group;
        int64_t       row_end   = row_start + rows_per_group;
        എങ്കിൽ (row_end > total_rows) {
            row_end = total_rows;
        }

        വേണ്ടി (int64_t row = row_start; row < row_end; ++row) {
            int64_t i1 = row % ne1;
            int64_t i2 = (row / ne1) % ne2;
            int64_t i3 = row / (ne1 * ne2);

            സ്ഥിരം ദശാംശം * src_row = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0_data + i1 * snb1 + i2 * snb2 + i3 * snb3);
            ദശാംശം *       dst_row = (ദശാംശം *) ((അക്ഷരം *) dst_data + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            ദശാംശം acc = 0.0f;
            വേണ്ടി (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_row[i0];
                dst_row[i0] = acc;
            }
        }
    }

    തിരികെ 0;
}
