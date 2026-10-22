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

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <stdbool.h>
#અન્તર્ભાવ <stdint.h>

સંરચના ggml_et_cont_params {
    સંરચના ggml_tensor src0;  // F32 input tensor (non-contiguous)
    સંરચના ggml_tensor dst;   // F32 output tensor (contiguous)
};

// Vectorized copy with scalar tail
સ્થિર અંતઃરેખ શૂન્ય vec_copy_f32(દશાંશ * dst, અચલ દશાંશ * src, int32_t n) {
    int32_t       i       = 0;
    અચલ int32_t vec_end = (n / 8) * 8;
    માટે (; i < vec_end; i += 8) {
        __asm__ અસ્થિર(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(દશાંશ (*)[8]) & dst[i])
            : [s] "m"(*(અચલ દશાંશ (*)[8]) & src[i])
            : "f10");
    }
    માટે (; i < n; i++) {
        dst[i] = src[i];
    }
}

// Scalar copy
સ્થિર અંતઃરેખ શૂન્ય scalar_copy_f32(દશાંશ * dst, અચલ દશાંશ * src, int32_t n) {
    માટે (int32_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

// static inline size_t tensor_bytes(const struct ggml_tensor *t) {
//     return (size_t)t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
// }

પૂર્ણાંક entry_point(સંરચના ggml_et_cont_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    જો (!kernel_env) {
        પાછા -1;
    }

    પૂર્ણાંક thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    પૂર્ણાંક અંક_threads = get_અંક_threads(kernel_env->shire_mask);

    જો (thread_id < 0) {
        પાછા 0;
    }

    જો (params == 0 || ((uint64_t) params & 0x7) != 0) {
        પાછા -1;
    }

    સંરચના ggml_tensor * src0 = &params->src0;
    સંરચના ggml_tensor * dst  = &params->dst;

    જો (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા -1;
    }

    દશાંશ * src0_data = (દશાંશ *) src0->data;
    દશાંશ * dst_data  = (દશાંશ *) dst->data;
    // evict_region_past_l2(src0_data, tensor_bytes(src0));

    જો (!src0_data || !dst_data) {
        પાછા -1;
    }

    અચલ int64_t ne00 = src0->ne[0];
    અચલ int64_t ne01 = src0->ne[1];
    અચલ int64_t ne02 = src0->ne[2];
    અચલ int64_t ne03 = src0->ne[3];

    અચલ int64_t nb00 = src0->nb[0];
    અચલ int64_t nb01 = src0->nb[1];
    અચલ int64_t nb02 = src0->nb[2];
    અચલ int64_t nb03 = src0->nb[3];

    અચલ int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    જો (total_elements == 0) {
        પાછા 0;
    }

    અચલ bool src_contiguous = ggml_tensor_is_contiguous(src0, 4);

    //==========================================================================
    // Fast path: src is contiguous: flat vectorized copy by cache lines
    //==========================================================================
    જો (src_contiguous) {
        અચલ int64_t elems_per_cl = 16;
        અચલ int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

        અચલ int64_t cl_per_thread = (total_cl + અંક_threads - 1) / અંક_threads;
        અચલ int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        જો (cl_end > total_cl) {
            cl_end = total_cl;
        }
        જો (cl_start >= total_cl) {
            પાછા 0;
        }

        અચલ int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        જો (ee > total_elements) {
            ee = total_elements;
        }

        vec_copy_f32(dst_data + es, src0_data + es, (int32_t) (ee - es));
        પાછા 0;
    }

    //==========================================================================
    // Non-contiguous paths: require nb00==4 (dim 0 contiguous in src)
    //==========================================================================
    જો (nb00 != 4) {
        // Fully non-contiguous scalar fallback — distribute by cache lines
        અચલ int64_t elems_per_cl = 16;
        અચલ int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

        અચલ int64_t cl_per_thread = (total_cl + અંક_threads - 1) / અંક_threads;
        અચલ int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        જો (cl_end > total_cl) {
            cl_end = total_cl;
        }
        જો (cl_start >= total_cl) {
            પાછા 0;
        }

        અચલ int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        જો (ee > total_elements) {
            ee = total_elements;
        }

        માટે (int64_t idx = es; idx < ee; idx++) {
            અચલ int64_t i00  = idx % ne00;
            અચલ int64_t rem1 = idx / ne00;
            અચલ int64_t i01  = rem1 % ne01;
            અચલ int64_t rem2 = rem1 / ne01;
            અચલ int64_t i02  = rem2 % ne02;
            અચલ int64_t i03  = rem2 / ne02;

            અચલ દશાંશ * sp =
                (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_data[idx] = *sp;
        }
        પાછા 0;
    }

    // nb00 == 4 from here: dim 0 is contiguous in src

    //==========================================================================
    // Aligned path: ne00 % 16 == 0: rows are cache-line aligned, distribute rows
    //==========================================================================
    જો (ne00 % 16 == 0) {
        અચલ int64_t total_rows      = ne01 * ne02 * ne03;
        અચલ int64_t rows_per_thread = (total_rows + અંક_threads - 1) / અંક_threads;
        અચલ int64_t start_row       = thread_id * rows_per_thread;
        અચલ int64_t end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;

        જો (start_row >= total_rows) {
            પાછા 0;
        }

        માટે (int64_t ir = start_row; ir < end_row; ir++) {
            અચલ int64_t i03 = ir / (ne02 * ne01);
            અચલ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            અચલ int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            અચલ દશાંશ * src_row = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
            દશાંશ *       dst_row = dst_data + ir * ne00;

            vec_copy_f32(dst_row, src_row, (int32_t) ne00);
        }
        પાછા 0;
    }

    //==========================================================================
    // Unaligned path: ne00 % 16 != 0, nb00 == 4
    // Distribute cache-line-aligned chunks of dst, handle partial rows at edges
    //==========================================================================
    {
        અચલ int64_t elems_per_cl = 16;
        અચલ int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

        અચલ int64_t cl_per_thread = (total_cl + અંક_threads - 1) / અંક_threads;
        અચલ int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        જો (cl_end > total_cl) {
            cl_end = total_cl;
        }
        જો (cl_start >= total_cl) {
            પાછા 0;
        }

        અચલ int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        જો (ee > total_elements) {
            ee = total_elements;
        }

        int64_t pos = es;

        // Compute starting row coordinates
        int64_t row_idx = pos / ne00;
        int64_t col     = pos % ne00;

        જ્યારે (pos < ee) {
            // Decompose row_idx -> (i01, i02, i03)
            અચલ int64_t i03 = row_idx / (ne02 * ne01);
            અચલ int64_t i02 = (row_idx - i03 * ne02 * ne01) / ne01;
            અચલ int64_t i01 = row_idx - i03 * ne02 * ne01 - i02 * ne01;

            અચલ દશાંશ * src_row = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);

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

    પાછા 0;
}
