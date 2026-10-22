//******************************************************************************
// Diag F32 Kernel
// Creates a diagonal matrix from a 1D vector.
// dst[i][j] = (i == j) ? src0[i] : 0.0f
//
// src0: [N, 1, ne2, ne3]   (1D vector per batch)
// dst:  [N, N, ne2, ne3]   (diagonal matrix per batch)
//******************************************************************************

#অন্তর্ভাৱ "ggml_tensor.h"
#অন্তর্ভাৱ "platform.h"

#অন্তর্ভাৱ <stdint.h>

সংরচনা ggml_et_diag_params {
    সংরচনা ggml_tensor src0;  // F32 input vector
    সংরচনা ggml_tensor dst;   // F32 output diagonal matrix
};

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_diag_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ঘূৰাই -1;
    }

    পূর্ণাঙ্ক thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণাঙ্ক অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ঘূৰাই 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ঘূৰাই -1;
    }

    সংরচনা ggml_tensor * src0 = &params->src0;
    সংরচনা ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ঘূৰাই -1;
    }

    ভিন্ন * src0_data = (ভিন্ন *) src0->data;
    ভিন্ন * dst_data  = (ভিন্ন *) dst->data;

    যদি (!src0_data || !dst_data) {
        ঘূৰাই -1;
    }

    ধ্রুৱ int64_t ne0 = dst->ne[0];  // N (row width = column count)
    ধ্রুৱ int64_t ne1 = dst->ne[1];  // N (number of rows)
    ধ্রুৱ int64_t ne2 = dst->ne[2];
    ধ্রুৱ int64_t ne3 = dst->ne[3];

    ধ্রুৱ size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ধ্রুৱ size_t nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Total rows across all batches — parallelize over these
    ধ্রুৱ int64_t total_rows = ne1 * ne2 * ne3;

    // Prepare zero vector for SIMD zeroing
    ভিন্ন zero = 0.0f;
    __asm__ চঞ্চল("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

    বাবে (int64_t row = thread_id; row < total_rows; row += অঙ্ক_threads) {
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);

        ভিন্ন * dst_row = (ভিন্ন *) ((ৱর্ণ *) dst_data + i1 * nb1 + i2 * nb2 + i3 * nb3);

        // Zero the entire row with SIMD
        int64_t       i0      = 0;
        ধ্রুৱ int64_t vec_end = (ne0 / 8) * 8;
        বাবে (; i0 < vec_end; i0 += 8) {
            __asm__ চঞ্চল("fsw.ps f10, %[d]\n" : [d] "=m"(*(ভিন্ন (*)[8]) & dst_row[i0])::"f10");
        }
        বাবে (; i0 < ne0; i0++) {
            dst_row[i0] = 0.0f;
        }

        // Place the diagonal element: dst[i1][i1] = src0[i1]
        ধ্রুৱ ভিন্ন * src_ptr = (ধ্রুৱ ভিন্ন *) ((ধ্রুৱ ৱর্ণ *) src0_data + i2 * nb02 + i3 * nb03);
        dst_row[i1]           = src_ptr[i1];
    }

    ঘূৰাই 0;
}
