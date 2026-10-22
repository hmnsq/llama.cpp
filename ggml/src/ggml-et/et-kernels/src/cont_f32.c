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

#समावेश "ggml_tensor.h"
#समावेश "platform.h"

#समावेश <stdbool.h>
#समावेश <stdint.h>

काष्ठा ggml_et_cont_params {
    काष्ठा ggml_tensor src0;  // F32 input tensor (non-contiguous)
    काष्ठा ggml_tensor dst;   // F32 output tensor (contiguous)
};

// Vectorized copy with scalar tail
अटल अंतरभूत व्योम vec_copy_f32(भग्न * dst, स्थिर भग्न * src, int32_t n) {
    int32_t       i       = 0;
    स्थिर int32_t vec_end = (n / 8) * 8;
    क्रम (; i < vec_end; i += 8) {
        __asm__ अस्थिर(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(भग्न (*)[8]) & dst[i])
            : [s] "m"(*(स्थिर भग्न (*)[8]) & src[i])
            : "f10");
    }
    क्रम (; i < n; i++) {
        dst[i] = src[i];
    }
}

// Scalar copy
अटल अंतरभूत व्योम scalar_copy_f32(भग्न * dst, स्थिर भग्न * src, int32_t n) {
    क्रम (int32_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

// static inline size_t tensor_bytes(const struct ggml_tensor *t) {
//     return (size_t)t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
// }

पूर्णांक entry_point(काष्ठा ggml_et_cont_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस 0;
    }

    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;
    }

    काष्ठा ggml_tensor * src0 = &params->src0;
    काष्ठा ggml_tensor * dst  = &params->dst;

    अगर (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        वापस -1;
    }

    भग्न * src0_तथ्य = (भग्न *) src0->तथ्य;
    भग्न * dst_तथ्य  = (भग्न *) dst->तथ्य;
    // evict_region_past_l2(src0_data, tensor_bytes(src0));

    अगर (!src0_तथ्य || !dst_तथ्य) {
        वापस -1;
    }

    स्थिर int64_t ne00 = src0->ne[0];
    स्थिर int64_t ne01 = src0->ne[1];
    स्थिर int64_t ne02 = src0->ne[2];
    स्थिर int64_t ne03 = src0->ne[3];

    स्थिर int64_t nb00 = src0->nb[0];
    स्थिर int64_t nb01 = src0->nb[1];
    स्थिर int64_t nb02 = src0->nb[2];
    स्थिर int64_t nb03 = src0->nb[3];

    स्थिर int64_t योग_elements = ne00 * ne01 * ne02 * ne03;

    अगर (योग_elements == 0) {
        वापस 0;
    }

    स्थिर bool src_contiguous = ggml_tensor_is_contiguous(src0, 4);

    //==========================================================================
    // Fast path: src is contiguous: flat vectorized copy by cache lines
    //==========================================================================
    अगर (src_contiguous) {
        स्थिर int64_t elems_per_cl = 16;
        स्थिर int64_t योग_cl     = (योग_elements + elems_per_cl - 1) / elems_per_cl;

        स्थिर int64_t cl_per_thread = (योग_cl + अंक_threads - 1) / अंक_threads;
        स्थिर int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        अगर (cl_end > योग_cl) {
            cl_end = योग_cl;
        }
        अगर (cl_start >= योग_cl) {
            वापस 0;
        }

        स्थिर int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        अगर (ee > योग_elements) {
            ee = योग_elements;
        }

        vec_copy_f32(dst_तथ्य + es, src0_तथ्य + es, (int32_t) (ee - es));
        वापस 0;
    }

    //==========================================================================
    // Non-contiguous paths: require nb00==4 (dim 0 contiguous in src)
    //==========================================================================
    अगर (nb00 != 4) {
        // Fully non-contiguous scalar fallback — distribute by cache lines
        स्थिर int64_t elems_per_cl = 16;
        स्थिर int64_t योग_cl     = (योग_elements + elems_per_cl - 1) / elems_per_cl;

        स्थिर int64_t cl_per_thread = (योग_cl + अंक_threads - 1) / अंक_threads;
        स्थिर int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        अगर (cl_end > योग_cl) {
            cl_end = योग_cl;
        }
        अगर (cl_start >= योग_cl) {
            वापस 0;
        }

        स्थिर int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        अगर (ee > योग_elements) {
            ee = योग_elements;
        }

        क्रम (int64_t idx = es; idx < ee; idx++) {
            स्थिर int64_t i00  = idx % ne00;
            स्थिर int64_t rem1 = idx / ne00;
            स्थिर int64_t i01  = rem1 % ne01;
            स्थिर int64_t rem2 = rem1 / ne01;
            स्थिर int64_t i02  = rem2 % ne02;
            स्थिर int64_t i03  = rem2 / ne02;

            स्थिर भग्न * sp =
                (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_तथ्य[idx] = *sp;
        }
        वापस 0;
    }

    // nb00 == 4 from here: dim 0 is contiguous in src

    //==========================================================================
    // Aligned path: ne00 % 16 == 0: rows are cache-line aligned, distribute rows
    //==========================================================================
    अगर (ne00 % 16 == 0) {
        स्थिर int64_t योग_rows      = ne01 * ne02 * ne03;
        स्थिर int64_t rows_per_thread = (योग_rows + अंक_threads - 1) / अंक_threads;
        स्थिर int64_t start_row       = thread_id * rows_per_thread;
        स्थिर int64_t end_row = (start_row + rows_per_thread < योग_rows) ? (start_row + rows_per_thread) : योग_rows;

        अगर (start_row >= योग_rows) {
            वापस 0;
        }

        क्रम (int64_t ir = start_row; ir < end_row; ir++) {
            स्थिर int64_t i03 = ir / (ne02 * ne01);
            स्थिर int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            स्थिर int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            स्थिर भग्न * src_row = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i01 * nb01 + i02 * nb02 + i03 * nb03);
            भग्न *       dst_row = dst_तथ्य + ir * ne00;

            vec_copy_f32(dst_row, src_row, (int32_t) ne00);
        }
        वापस 0;
    }

    //==========================================================================
    // Unaligned path: ne00 % 16 != 0, nb00 == 4
    // Distribute cache-line-aligned chunks of dst, handle partial rows at edges
    //==========================================================================
    {
        स्थिर int64_t elems_per_cl = 16;
        स्थिर int64_t योग_cl     = (योग_elements + elems_per_cl - 1) / elems_per_cl;

        स्थिर int64_t cl_per_thread = (योग_cl + अंक_threads - 1) / अंक_threads;
        स्थिर int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        अगर (cl_end > योग_cl) {
            cl_end = योग_cl;
        }
        अगर (cl_start >= योग_cl) {
            वापस 0;
        }

        स्थिर int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        अगर (ee > योग_elements) {
            ee = योग_elements;
        }

        int64_t pos = es;

        // Compute starting row coordinates
        int64_t row_idx = pos / ne00;
        int64_t col     = pos % ne00;

        जबतक (pos < ee) {
            // Decompose row_idx -> (i01, i02, i03)
            स्थिर int64_t i03 = row_idx / (ne02 * ne01);
            स्थिर int64_t i02 = (row_idx - i03 * ne02 * ne01) / ne01;
            स्थिर int64_t i01 = row_idx - i03 * ne02 * ne01 - i02 * ne01;

            स्थिर भग्न * src_row = (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + i01 * nb01 + i02 * nb02 + i03 * nb03);

            // How many elements left in this row and in our chunk
            int64_t row_remaining   = ne00 - col;
            int64_t chunk_remaining = ee - pos;
            int32_t n               = (int32_t) (row_remaining < chunk_remaining ? row_remaining : chunk_remaining);

            vec_copy_f32(dst_तथ्य + pos, src_row + col, n);

            pos += n;
            col = 0;  // subsequent rows start at column 0
            row_idx++;
        }
    }

    वापस 0;
}
