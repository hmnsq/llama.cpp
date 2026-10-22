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

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdint.h>

संरचना ggml_et_cont_params {
    संरचना ggml_tensor src0;  // F32 input tensor (non-contiguous)
    संरचना ggml_tensor dst;   // F32 output tensor (contiguous)
};

// Vectorized copy with scalar tail
स्थिर अन्तःस्थ शून्य vec_copy_f32(भिन्न * dst, ध्रुव भिन्न * src, int32_t n) {
    int32_t       i       = 0;
    ध्रुव int32_t vec_end = (n / 8) * 8;
    प्रति (; i < vec_end; i += 8) {
        __asm__ चञ्चल(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(भिन्न (*)[8]) & dst[i])
            : [s] "m"(*(ध्रुव भिन्न (*)[8]) & src[i])
            : "f10");
    }
    प्रति (; i < n; i++) {
        dst[i] = src[i];
    }
}

// Scalar copy
स्थिर अन्तःस्थ शून्य scalar_copy_f32(भिन्न * dst, ध्रुव भिन्न * src, int32_t n) {
    प्रति (int32_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

// static inline size_t tensor_bytes(const struct ggml_tensor *t) {
//     return (size_t)t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
// }

पूर्णाङ्क entry_point(संरचना ggml_et_cont_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    यदि (thread_id < 0) {
        प्रत्यागम 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    भिन्न * src0_data = (भिन्न *) src0->data;
    भिन्न * dst_data  = (भिन्न *) dst->data;
    // evict_region_past_l2(src0_data, tensor_bytes(src0));

    यदि (!src0_data || !dst_data) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t ne00 = src0->ne[0];
    ध्रुव int64_t ne01 = src0->ne[1];
    ध्रुव int64_t ne02 = src0->ne[2];
    ध्रुव int64_t ne03 = src0->ne[3];

    ध्रुव int64_t nb00 = src0->nb[0];
    ध्रुव int64_t nb01 = src0->nb[1];
    ध्रुव int64_t nb02 = src0->nb[2];
    ध्रुव int64_t nb03 = src0->nb[3];

    ध्रुव int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    यदि (total_elements == 0) {
        प्रत्यागम 0;
    }

    ध्रुव bool src_contiguous = ggml_tensor_is_contiguous(src0, 4);

    //==========================================================================
    // Fast path: src is contiguous: flat vectorized copy by cache lines
    //==========================================================================
    यदि (src_contiguous) {
        ध्रुव int64_t elems_per_cl = 16;
        ध्रुव int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

        ध्रुव int64_t cl_per_thread = (total_cl + अंक_threads - 1) / अंक_threads;
        ध्रुव int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        यदि (cl_end > total_cl) {
            cl_end = total_cl;
        }
        यदि (cl_start >= total_cl) {
            प्रत्यागम 0;
        }

        ध्रुव int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        यदि (ee > total_elements) {
            ee = total_elements;
        }

        vec_copy_f32(dst_data + es, src0_data + es, (int32_t) (ee - es));
        प्रत्यागम 0;
    }

    //==========================================================================
    // Non-contiguous paths: require nb00==4 (dim 0 contiguous in src)
    //==========================================================================
    यदि (nb00 != 4) {
        // Fully non-contiguous scalar fallback — distribute by cache lines
        ध्रुव int64_t elems_per_cl = 16;
        ध्रुव int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

        ध्रुव int64_t cl_per_thread = (total_cl + अंक_threads - 1) / अंक_threads;
        ध्रुव int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        यदि (cl_end > total_cl) {
            cl_end = total_cl;
        }
        यदि (cl_start >= total_cl) {
            प्रत्यागम 0;
        }

        ध्रुव int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        यदि (ee > total_elements) {
            ee = total_elements;
        }

        प्रति (int64_t idx = es; idx < ee; idx++) {
            ध्रुव int64_t i00  = idx % ne00;
            ध्रुव int64_t rem1 = idx / ne00;
            ध्रुव int64_t i01  = rem1 % ne01;
            ध्रुव int64_t rem2 = rem1 / ne01;
            ध्रुव int64_t i02  = rem2 % ne02;
            ध्रुव int64_t i03  = rem2 / ne02;

            ध्रुव भिन्न * sp =
                (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_data[idx] = *sp;
        }
        प्रत्यागम 0;
    }

    // nb00 == 4 from here: dim 0 is contiguous in src

    //==========================================================================
    // Aligned path: ne00 % 16 == 0: rows are cache-line aligned, distribute rows
    //==========================================================================
    यदि (ne00 % 16 == 0) {
        ध्रुव int64_t total_rows      = ne01 * ne02 * ne03;
        ध्रुव int64_t rows_per_thread = (total_rows + अंक_threads - 1) / अंक_threads;
        ध्रुव int64_t start_row       = thread_id * rows_per_thread;
        ध्रुव int64_t end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;

        यदि (start_row >= total_rows) {
            प्रत्यागम 0;
        }

        प्रति (int64_t ir = start_row; ir < end_row; ir++) {
            ध्रुव int64_t i03 = ir / (ne02 * ne01);
            ध्रुव int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ध्रुव int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            ध्रुव भिन्न * src_row = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
            भिन्न *       dst_row = dst_data + ir * ne00;

            vec_copy_f32(dst_row, src_row, (int32_t) ne00);
        }
        प्रत्यागम 0;
    }

    //==========================================================================
    // Unaligned path: ne00 % 16 != 0, nb00 == 4
    // Distribute cache-line-aligned chunks of dst, handle partial rows at edges
    //==========================================================================
    {
        ध्रुव int64_t elems_per_cl = 16;
        ध्रुव int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

        ध्रुव int64_t cl_per_thread = (total_cl + अंक_threads - 1) / अंक_threads;
        ध्रुव int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        यदि (cl_end > total_cl) {
            cl_end = total_cl;
        }
        यदि (cl_start >= total_cl) {
            प्रत्यागम 0;
        }

        ध्रुव int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        यदि (ee > total_elements) {
            ee = total_elements;
        }

        int64_t pos = es;

        // Compute starting row coordinates
        int64_t row_idx = pos / ne00;
        int64_t col     = pos % ne00;

        यावत् (pos < ee) {
            // Decompose row_idx -> (i01, i02, i03)
            ध्रुव int64_t i03 = row_idx / (ne02 * ne01);
            ध्रुव int64_t i02 = (row_idx - i03 * ne02 * ne01) / ne01;
            ध्रुव int64_t i01 = row_idx - i03 * ne02 * ne01 - i02 * ne01;

            ध्रुव भिन्न * src_row = (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);

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

    प्रत्यागम 0;
}
