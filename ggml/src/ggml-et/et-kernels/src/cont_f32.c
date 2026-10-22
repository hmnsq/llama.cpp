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
स्थिर अन्तःस्थ शून्य vec_copy_f32(भिन्न * dst, अचर भिन्न * src, int32_t n) {
    int32_t       i       = 0;
    अचर int32_t vec_end = (n / 8) * 8;
    लागि (; i < vec_end; i += 8) {
        __asm__ चञ्चल(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(भिन्न (*)[8]) & dst[i])
            : [s] "m"(*(अचर भिन्न (*)[8]) & src[i])
            : "f10");
    }
    लागि (; i < n; i++) {
        dst[i] = src[i];
    }
}

// Scalar copy
स्थिर अन्तःस्थ शून्य scalar_copy_f32(भिन्न * dst, अचर भिन्न * src, int32_t n) {
    लागि (int32_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

// static inline size_t tensor_bytes(const struct ggml_tensor *t) {
//     return (size_t)t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
// }

पूर्णाङ्क entry_point(संरचना ggml_et_cont_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        फर्काऊ -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णाङ्क अङ्क_threads = get_अङ्क_threads(kernel_env->shire_mask);

    यदि (thread_id < 0) {
        फर्काऊ 0;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        फर्काऊ -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        फर्काऊ -1;
    }

    भिन्न * src0_data = (भिन्न *) src0->data;
    भिन्न * dst_data  = (भिन्न *) dst->data;
    // evict_region_past_l2(src0_data, tensor_bytes(src0));

    यदि (!src0_data || !dst_data) {
        फर्काऊ -1;
    }

    अचर int64_t ne00 = src0->ne[0];
    अचर int64_t ne01 = src0->ne[1];
    अचर int64_t ne02 = src0->ne[2];
    अचर int64_t ne03 = src0->ne[3];

    अचर int64_t nb00 = src0->nb[0];
    अचर int64_t nb01 = src0->nb[1];
    अचर int64_t nb02 = src0->nb[2];
    अचर int64_t nb03 = src0->nb[3];

    अचर int64_t total_elements = ne00 * ne01 * ne02 * ne03;

    यदि (total_elements == 0) {
        फर्काऊ 0;
    }

    अचर bool src_contiguous = ggml_tensor_is_contiguous(src0, 4);

    //==========================================================================
    // Fast path: src is contiguous: flat vectorized copy by cache lines
    //==========================================================================
    यदि (src_contiguous) {
        अचर int64_t elems_per_cl = 16;
        अचर int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

        अचर int64_t cl_per_thread = (total_cl + अङ्क_threads - 1) / अङ्क_threads;
        अचर int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        यदि (cl_end > total_cl) {
            cl_end = total_cl;
        }
        यदि (cl_start >= total_cl) {
            फर्काऊ 0;
        }

        अचर int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        यदि (ee > total_elements) {
            ee = total_elements;
        }

        vec_copy_f32(dst_data + es, src0_data + es, (int32_t) (ee - es));
        फर्काऊ 0;
    }

    //==========================================================================
    // Non-contiguous paths: require nb00==4 (dim 0 contiguous in src)
    //==========================================================================
    यदि (nb00 != 4) {
        // Fully non-contiguous scalar fallback — distribute by cache lines
        अचर int64_t elems_per_cl = 16;
        अचर int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

        अचर int64_t cl_per_thread = (total_cl + अङ्क_threads - 1) / अङ्क_threads;
        अचर int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        यदि (cl_end > total_cl) {
            cl_end = total_cl;
        }
        यदि (cl_start >= total_cl) {
            फर्काऊ 0;
        }

        अचर int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        यदि (ee > total_elements) {
            ee = total_elements;
        }

        लागि (int64_t idx = es; idx < ee; idx++) {
            अचर int64_t i00  = idx % ne00;
            अचर int64_t rem1 = idx / ne00;
            अचर int64_t i01  = rem1 % ne01;
            अचर int64_t rem2 = rem1 / ne01;
            अचर int64_t i02  = rem2 % ne02;
            अचर int64_t i03  = rem2 / ne02;

            अचर भिन्न * sp =
                (अचर भिन्न *) ((अचर वर्ण *) src0_data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_data[idx] = *sp;
        }
        फर्काऊ 0;
    }

    // nb00 == 4 from here: dim 0 is contiguous in src

    //==========================================================================
    // Aligned path: ne00 % 16 == 0: rows are cache-line aligned, distribute rows
    //==========================================================================
    यदि (ne00 % 16 == 0) {
        अचर int64_t total_rows      = ne01 * ne02 * ne03;
        अचर int64_t rows_per_thread = (total_rows + अङ्क_threads - 1) / अङ्क_threads;
        अचर int64_t start_row       = thread_id * rows_per_thread;
        अचर int64_t end_row = (start_row + rows_per_thread < total_rows) ? (start_row + rows_per_thread) : total_rows;

        यदि (start_row >= total_rows) {
            फर्काऊ 0;
        }

        लागि (int64_t ir = start_row; ir < end_row; ir++) {
            अचर int64_t i03 = ir / (ne02 * ne01);
            अचर int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            अचर int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            अचर भिन्न * src_row = (अचर भिन्न *) ((अचर वर्ण *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);
            भिन्न *       dst_row = dst_data + ir * ne00;

            vec_copy_f32(dst_row, src_row, (int32_t) ne00);
        }
        फर्काऊ 0;
    }

    //==========================================================================
    // Unaligned path: ne00 % 16 != 0, nb00 == 4
    // Distribute cache-line-aligned chunks of dst, handle partial rows at edges
    //==========================================================================
    {
        अचर int64_t elems_per_cl = 16;
        अचर int64_t total_cl     = (total_elements + elems_per_cl - 1) / elems_per_cl;

        अचर int64_t cl_per_thread = (total_cl + अङ्क_threads - 1) / अङ्क_threads;
        अचर int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        यदि (cl_end > total_cl) {
            cl_end = total_cl;
        }
        यदि (cl_start >= total_cl) {
            फर्काऊ 0;
        }

        अचर int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        यदि (ee > total_elements) {
            ee = total_elements;
        }

        int64_t pos = es;

        // Compute starting row coordinates
        int64_t row_idx = pos / ne00;
        int64_t col     = pos % ne00;

        जबसम्म (pos < ee) {
            // Decompose row_idx -> (i01, i02, i03)
            अचर int64_t i03 = row_idx / (ne02 * ne01);
            अचर int64_t i02 = (row_idx - i03 * ne02 * ne01) / ne01;
            अचर int64_t i01 = row_idx - i03 * ne02 * ne01 - i02 * ne01;

            अचर भिन्न * src_row = (अचर भिन्न *) ((अचर वर्ण *) src0_data + i01 * nb01 + i02 * nb02 + i03 * nb03);

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

    फर्काऊ 0;
}
