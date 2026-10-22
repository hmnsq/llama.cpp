//******************************************************************************
// Bare Metal CONT F32 Kernel
// Converts non-contiguous tensors to contiguous memory layout
//
// Fast path:    src contiguous: flat vectorized copy by cache lines
// Aligned path: nb00==4 and ne00 % 16 == 0: distribute rows, no coherency issue
// Unaligned:    nb00==4 and ne00 not aligned: distribute by cache lines,
//               reverse-compute src coords, handle partial rows at boundaries
// Fallback:     nb00 != 4: scalar per-element
//******************************************************************************

#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <stdbool.h>
#അന്തര്ഭാവ <stdint.h>

ഘടന ggml_et_cont_params {
    ഘടന ggml_tensor src0;  // F32 input tensor (non-contiguous)
    ഘടന ggml_tensor dst;   // F32 output tensor (contiguous)
};

// Vectorized copy with scalar tail
സ്ഥിതം വരിയിൽ ശൂന്യം vec_copy_f32(ദശാംശം * dst, സ്ഥിരം ദശാംശം * src, int32_t n) {
    int32_t       i       = 0;
    സ്ഥിരം int32_t vec_end = (n / 8) * 8;
    വേണ്ടി (; i < vec_end; i += 8) {
        __asm__ അസ്ഥിര(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(ദശാംശം (*)[8]) & dst[i])
            : [s] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) & src[i])
            : "f10");
    }
    വേണ്ടി (; i < n; i++) {
        dst[i] = src[i];
    }
}

// Scalar copy
സ്ഥിതം വരിയിൽ ശൂന്യം scalar_copy_f32(ദശാംശം * dst, സ്ഥിരം ദശാംശം * src, int32_t n) {
    വേണ്ടി (int32_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

// static inline size_t tensor_bytes(const struct ggml_tensor *t) {
//     return (size_t)t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
// }

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_cont_params * params, ശൂന്യം * env) {
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
    // evict_region_past_l2(src0_data, tensor_bytes(src0));

    എങ്കിൽ (!src0_data || !dst_data) {
        തിരികെ -1;
    }

    സ്ഥിരം int64_t ne00 = src0->ne[0];
    സ്ഥിരം int64_t ne01 = src0->ne[1];
    സ്ഥിരം int64_t ne02 = src0->ne[2];
    സ്ഥിരം int64_t ne03 = src0->ne[3];

    സ്ഥിരം int64_t nb00 = src0->nb[0];
    സ്ഥിരം int64_t nb01 = src0->nb[1];
    സ്ഥിരം int64_t nb02 = src0->nb[2];
    സ്ഥിരം int64_t nb03 = src0->nb[3];

    സ്ഥിരം int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    എങ്കിൽ (total_elements == 0) {
        തിരികെ 0;
    }

    സ്ഥിരം bool src_contiguous = ggml_tensor_is_contiguous(src0, 4);

    //==========================================================================
    // Fast path: src is contiguous: flat vectorized copy by cache lines
    //==========================================================================
    എങ്കിൽ (src_contiguous) {
        സ്ഥിരം int64_t elems_per_cl = 16;
        സ്ഥിരം int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

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

        vec_copy_f32(dst_data + es, src0_data + es, (int32_t) (ee - es));
        തിരികെ 0;
    }

    //==========================================================================
    // Non-contiguous paths: require nb00==4 (dim 0 contiguous in src)
    //==========================================================================
    എങ്കിൽ (nb00 != 4) {
        // Fully non-contiguous scalar fallback — distribute by cache lines
        സ്ഥിരം int64_t elems_per_cl = 16;
        സ്ഥിരം int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

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

        വേണ്ടി (int64_t idx = es; idx < ee; idx++) {
            സ്ഥിരം int64_t i00  = idx % ne00;
            സ്ഥിരം int64_t rem1 = idx / ne00;
            സ്ഥിരം int64_t i01  = rem1 % ne01;
            സ്ഥിരം int64_t rem2 = rem1 / ne01;
            സ്ഥിരം int64_t i02  = rem2 % ne02;
            സ്ഥിരം int64_t i03  = rem2 / ne02;

            സ്ഥിരം ദശാംശം * sp =
                (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0_data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_data[idx] = *sp;
        }
        തിരികെ 0;
    }

    // nb00 == 4 from here: dim 0 is contiguous in src

    //==========================================================================
    // Aligned path: ne00 % 16 == 0: rows are cache-line aligned, distribute rows
    //==========================================================================
    എങ്കിൽ (ne00 % 16 == 0) {
        സ്ഥിരം int64_t total_rows      = ne01 * ne02 * ne03;
        സ്ഥിരം int64_t rows_per_thread = (total_rows + അക്കം_threads - 1) / അക്കം_threads;
        സ്ഥിരം int64_t start_row       = thread_id * rows_per_thread;
        സ്ഥിരം int64_t end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;

        എങ്കിൽ (start_row >= total_rows) {
            തിരികെ 0;
        }

        വേണ്ടി (int64_t ir = start_row; ir < end_row; ir++) {
            സ്ഥിരം int64_t i03 = ir / (ne02 * ne01);
            സ്ഥിരം int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            സ്ഥിരം int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            സ്ഥിരം ദശാംശം * src_row = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
            ദശാംശം *       dst_row = dst_data + ir * ne00;

            vec_copy_f32(dst_row, src_row, (int32_t) ne00);
        }
        തിരികെ 0;
    }

    //==========================================================================
    // Unaligned path: ne00 % 16 != 0, nb00 == 4
    // Distribute cache-line-aligned chunks of dst, handle partial rows at edges
    //==========================================================================
    {
        സ്ഥിരം int64_t elems_per_cl = 16;
        സ്ഥിരം int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

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

        int64_t pos = es;

        // Compute starting row coordinates
        int64_t row_idx = pos / ne00;
        int64_t col     = pos % ne00;

        ആകുമ്പോൾ (pos < ee) {
            // Decompose row_idx -> (i01, i02, i03)
            സ്ഥിരം int64_t i03 = row_idx / (ne02 * ne01);
            സ്ഥിരം int64_t i02 = (row_idx - i03 * ne02 * ne01) / ne01;
            സ്ഥിരം int64_t i01 = row_idx - i03 * ne02 * ne01 - i02 * ne01;

            സ്ഥിരം ദശാംശം * src_row = (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);

            // How many elements left in this row and in our chunk
            int64_t row_remaining   = ne00 - col;
            int64_t chunk_remaining = ee - pos;
            int32_t n               = (int32_t) (row_remaining < chunk_remaining ? row_remaining : chunk_remaining);

            vec_copy_f32(dst_data + pos, src_row + col, n);

            pos += n;
            col = 0;  // subsequent rows start at column 0
            row_idx++;
        }
    }

    തിരികെ 0;
}
