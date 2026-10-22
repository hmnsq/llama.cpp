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

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব <stdint.h>

কাঠামো ggml_et_cont_params {
    কাঠামো ggml_tensor src0;  // F32 input tensor (non-contiguous)
    কাঠামো ggml_tensor dst;   // F32 output tensor (contiguous)
};

// Vectorized copy with scalar tail
স্থির অন্তঃরেখ শূন্য vec_copy_f32(দশমিক * dst, ধ্রুবক দশমিক * src, int32_t n) {
    int32_t       i       = 0;
    ধ্রুবক int32_t vec_end = (n / 8) * 8;
    জন্য (; i < vec_end; i += 8) {
        __asm__ অস্থির(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(দশমিক (*)[8]) & dst[i])
            : [s] "m"(*(ধ্রুবক দশমিক (*)[8]) & src[i])
            : "f10");
    }
    জন্য (; i < n; i++) {
        dst[i] = src[i];
    }
}

// Scalar copy
স্থির অন্তঃরেখ শূন্য scalar_copy_f32(দশমিক * dst, ধ্রুবক দশমিক * src, int32_t n) {
    জন্য (int32_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

// static inline size_t tensor_bytes(const struct ggml_tensor *t) {
//     return (size_t)t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
// }

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_cont_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ফেরত -1;
    }

    পূর্ণসংখ্যা thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণসংখ্যা অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ফেরত 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ফেরত -1;
    }

    কাঠামো ggml_tensor * src0 = &params->src0;
    কাঠামো ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ফেরত -1;
    }

    দশমিক * src0_তথ্য = (দশমিক *) src0->তথ্য;
    দশমিক * dst_তথ্য  = (দশমিক *) dst->তথ্য;
    // evict_region_past_l2(src0_data, tensor_bytes(src0));

    যদি (!src0_তথ্য || !dst_তথ্য) {
        ফেরত -1;
    }

    ধ্রুবক int64_t ne00 = src0->ne[0];
    ধ্রুবক int64_t ne01 = src0->ne[1];
    ধ্রুবক int64_t ne02 = src0->ne[2];
    ধ্রুবক int64_t ne03 = src0->ne[3];

    ধ্রুবক int64_t nb00 = src0->nb[0];
    ধ্রুবক int64_t nb01 = src0->nb[1];
    ধ্রুবক int64_t nb02 = src0->nb[2];
    ধ্রুবক int64_t nb03 = src0->nb[3];

    ধ্রুবক int64_t যোগ_elements = ne00 * ne01 * ne02 * ne03;

    যদি (যোগ_elements == 0) {
        ফেরত 0;
    }

    ধ্রুবক bool src_contiguous = ggml_tensor_is_contiguous(src0, 4);

    //==========================================================================
    // Fast path: src is contiguous: flat vectorized copy by cache lines
    //==========================================================================
    যদি (src_contiguous) {
        ধ্রুবক int64_t elems_per_cl = 16;
        ধ্রুবক int64_t যোগ_cl     = (যোগ_elements + elems_per_cl - 1) / elems_per_cl;

        ধ্রুবক int64_t cl_per_thread = (যোগ_cl + অঙ্ক_threads - 1) / অঙ্ক_threads;
        ধ্রুবক int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        যদি (cl_end > যোগ_cl) {
            cl_end = যোগ_cl;
        }
        যদি (cl_start >= যোগ_cl) {
            ফেরত 0;
        }

        ধ্রুবক int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        যদি (ee > যোগ_elements) {
            ee = যোগ_elements;
        }

        vec_copy_f32(dst_তথ্য + es, src0_তথ্য + es, (int32_t) (ee - es));
        ফেরত 0;
    }

    //==========================================================================
    // Non-contiguous paths: require nb00==4 (dim 0 contiguous in src)
    //==========================================================================
    যদি (nb00 != 4) {
        // Fully non-contiguous scalar fallback — distribute by cache lines
        ধ্রুবক int64_t elems_per_cl = 16;
        ধ্রুবক int64_t যোগ_cl     = (যোগ_elements + elems_per_cl - 1) / elems_per_cl;

        ধ্রুবক int64_t cl_per_thread = (যোগ_cl + অঙ্ক_threads - 1) / অঙ্ক_threads;
        ধ্রুবক int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        যদি (cl_end > যোগ_cl) {
            cl_end = যোগ_cl;
        }
        যদি (cl_start >= যোগ_cl) {
            ফেরত 0;
        }

        ধ্রুবক int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        যদি (ee > যোগ_elements) {
            ee = যোগ_elements;
        }

        জন্য (int64_t idx = es; idx < ee; idx++) {
            ধ্রুবক int64_t i00  = idx % ne00;
            ধ্রুবক int64_t rem1 = idx / ne00;
            ধ্রুবক int64_t i01  = rem1 % ne01;
            ধ্রুবক int64_t rem2 = rem1 / ne01;
            ধ্রুবক int64_t i02  = rem2 % ne02;
            ধ্রুবক int64_t i03  = rem2 / ne02;

            ধ্রুবক দশমিক * sp =
                (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src0_তথ্য + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            dst_তথ্য[idx] = *sp;
        }
        ফেরত 0;
    }

    // nb00 == 4 from here: dim 0 is contiguous in src

    //==========================================================================
    // Aligned path: ne00 % 16 == 0: rows are cache-line aligned, distribute rows
    //==========================================================================
    যদি (ne00 % 16 == 0) {
        ধ্রুবক int64_t যোগ_rows      = ne01 * ne02 * ne03;
        ধ্রুবক int64_t rows_per_thread = (যোগ_rows + অঙ্ক_threads - 1) / অঙ্ক_threads;
        ধ্রুবক int64_t start_row       = thread_id * rows_per_thread;
        ধ্রুবক int64_t end_row = (start_row + rows_per_thread < যোগ_rows) ? (start_row + rows_per_thread) : যোগ_rows;

        যদি (start_row >= যোগ_rows) {
            ফেরত 0;
        }

        জন্য (int64_t ir = start_row; ir < end_row; ir++) {
            ধ্রুবক int64_t i03 = ir / (ne02 * ne01);
            ধ্রুবক int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
            ধ্রুবক int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

            ধ্রুবক দশমিক * src_row = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src0_তথ্য + i01 * nb01 + i02 * nb02 + i03 * nb03);
            দশমিক *       dst_row = dst_তথ্য + ir * ne00;

            vec_copy_f32(dst_row, src_row, (int32_t) ne00);
        }
        ফেরত 0;
    }

    //==========================================================================
    // Unaligned path: ne00 % 16 != 0, nb00 == 4
    // Distribute cache-line-aligned chunks of dst, handle partial rows at edges
    //==========================================================================
    {
        ধ্রুবক int64_t elems_per_cl = 16;
        ধ্রুবক int64_t যোগ_cl     = (যোগ_elements + elems_per_cl - 1) / elems_per_cl;

        ধ্রুবক int64_t cl_per_thread = (যোগ_cl + অঙ্ক_threads - 1) / অঙ্ক_threads;
        ধ্রুবক int64_t cl_start      = thread_id * cl_per_thread;
        int64_t       cl_end        = cl_start + cl_per_thread;
        যদি (cl_end > যোগ_cl) {
            cl_end = যোগ_cl;
        }
        যদি (cl_start >= যোগ_cl) {
            ফেরত 0;
        }

        ধ্রুবক int64_t es = cl_start * elems_per_cl;
        int64_t       ee = cl_end * elems_per_cl;
        যদি (ee > যোগ_elements) {
            ee = যোগ_elements;
        }

        int64_t pos = es;

        // Compute starting row coordinates
        int64_t row_idx = pos / ne00;
        int64_t col     = pos % ne00;

        যখন (pos < ee) {
            // Decompose row_idx -> (i01, i02, i03)
            ধ্রুবক int64_t i03 = row_idx / (ne02 * ne01);
            ধ্রুবক int64_t i02 = (row_idx - i03 * ne02 * ne01) / ne01;
            ধ্রুবক int64_t i01 = row_idx - i03 * ne02 * ne01 - i02 * ne01;

            ধ্রুবক দশমিক * src_row = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src0_তথ্য + i01 * nb01 + i02 * nb02 + i03 * nb03);

            // How many elements left in this row and in our chunk
            int64_t row_remaining   = ne00 - col;
            int64_t chunk_remaining = ee - pos;
            int32_t n               = (int32_t) (row_remaining < chunk_remaining ? row_remaining : chunk_remaining);

            vec_copy_f32(dst_তথ্য + pos, src_row + col, n);

            pos += n;
            col = 0;  // subsequent rows start at column 0
            row_idx++;
        }
    }

    ফেরত 0;
}
