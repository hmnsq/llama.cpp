//******************************************************************************
// Diag F32 Kernel
// Creates a diagonal matrix from a 1D vector.
// dst[i][j] = (i == j) ? src0[i] : 0.0f
//
// src0: [N, 1, ne2, ne3]   (1D vector per batch)
// dst:  [N, N, ne2, ne3]   (diagonal matrix per batch)
//******************************************************************************

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

কাঠামো ggml_et_diag_params {
    কাঠামো ggml_tensor src0;  // F32 input vector
    কাঠামো ggml_tensor dst;   // F32 output diagonal matrix
};

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_diag_params * params, শূন্য * env) {
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

    যদি (!src0_তথ্য || !dst_তথ্য) {
        ফেরত -1;
    }

    ধ্রুবক int64_t ne0 = dst->ne[0];  // N (row width = column count)
    ধ্রুবক int64_t ne1 = dst->ne[1];  // N (number of rows)
    ধ্রুবক int64_t ne2 = dst->ne[2];
    ধ্রুবক int64_t ne3 = dst->ne[3];

    ধ্রুবক size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ধ্রুবক size_t nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Total rows across all batches — parallelize over these
    ধ্রুবক int64_t যোগ_rows = ne1 * ne2 * ne3;

    // Prepare zero vector for SIMD zeroing
    দশমিক zero = 0.0f;
    __asm__ অস্থির("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

    জন্য (int64_t row = thread_id; row < যোগ_rows; row += অঙ্ক_threads) {
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);

        দশমিক * dst_row = (দশমিক *) ((অক্ষর *) dst_তথ্য + i1 * nb1 + i2 * nb2 + i3 * nb3);

        // Zero the entire row with SIMD
        int64_t       i0      = 0;
        ধ্রুবক int64_t vec_end = (ne0 / 8) * 8;
        জন্য (; i0 < vec_end; i0 += 8) {
            __asm__ অস্থির("fsw.ps f10, %[d]\n" : [d] "=m"(*(দশমিক (*)[8]) & dst_row[i0])::"f10");
        }
        জন্য (; i0 < ne0; i0++) {
            dst_row[i0] = 0.0f;
        }

        // Place the diagonal element: dst[i1][i1] = src0[i1]
        ধ্রুবক দশমিক * src_ptr = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src0_তথ্য + i2 * nb02 + i3 * nb03);
        dst_row[i1]           = src_ptr[i1];
    }

    ফেরত 0;
}
