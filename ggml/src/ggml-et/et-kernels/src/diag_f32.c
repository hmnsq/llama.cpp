//******************************************************************************
// Diag F32 Kernel
// Creates a diagonal matrix from a 1D vector.
// dst[i][j] = (i == j) ? src0[i] : 0.0f
//
// src0: [N, 1, ne2, ne3]   (1D vector per batch)
// dst:  [N, N, ne2, ne3]   (diagonal matrix per batch)
//******************************************************************************

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <stdint.h>

கட்டமைப்பு ggml_et_diag_params {
    கட்டமைப்பு ggml_tensor src0;  // F32 input vector
    கட்டமைப்பு ggml_tensor dst;   // F32 output diagonal matrix
};

முழு entry_point(கட்டமைப்பு ggml_et_diag_params * params, வெறுமை * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    எனில் (!kernel_env) {
        திரும்பு -1;
    }

    முழு thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    முழு இலக்கம்_threads = get_இலக்கம்_threads(kernel_env->shire_mask);

    எனில் (thread_id < 0) {
        திரும்பு 0;
    }

    எனில் (params == 0 || ((uint64_t) params & 0x7) != 0) {
        திரும்பு -1;
    }

    கட்டமைப்பு ggml_tensor * src0 = &params->src0;
    கட்டமைப்பு ggml_tensor * dst  = &params->dst;

    எனில் (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        திரும்பு -1;
    }

    தசம * src0_data = (தசம *) src0->data;
    தசம * dst_data  = (தசம *) dst->data;

    எனில் (!src0_data || !dst_data) {
        திரும்பு -1;
    }

    மாறிலி int64_t ne0 = dst->ne[0];  // N (row width = column count)
    மாறிலி int64_t ne1 = dst->ne[1];  // N (number of rows)
    மாறிலி int64_t ne2 = dst->ne[2];
    மாறிலி int64_t ne3 = dst->ne[3];

    மாறிலி size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    மாறிலி size_t nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Total rows across all batches — parallelize over these
    மாறிலி int64_t total_rows = ne1 * ne2 * ne3;

    // Prepare zero vector for SIMD zeroing
    தசம zero = 0.0f;
    __asm__ நிலையற்ற("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

    க்கு (int64_t row = thread_id; row < total_rows; row += இலக்கம்_threads) {
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);

        தசம * dst_row = (தசம *) ((எழுத்து *) dst_data + i1 * nb1 + i2 * nb2 + i3 * nb3);

        // Zero the entire row with SIMD
        int64_t       i0      = 0;
        மாறிலி int64_t vec_end = (ne0 / 8) * 8;
        க்கு (; i0 < vec_end; i0 += 8) {
            __asm__ நிலையற்ற("fsw.ps f10, %[d]\n" : [d] "=m"(*(தசம (*)[8]) & dst_row[i0])::"f10");
        }
        க்கு (; i0 < ne0; i0++) {
            dst_row[i0] = 0.0f;
        }

        // Place the diagonal element: dst[i1][i1] = src0[i1]
        மாறிலி தசம * src_ptr = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0_data + i2 * nb02 + i3 * nb03);
        dst_row[i1]           = src_ptr[i1];
    }

    திரும்பு 0;
}
