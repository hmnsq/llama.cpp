//******************************************************************************
// Diag F32 Kernel
// Creates a diagonal matrix from a 1D vector.
// dst[i][j] = (i == j) ? src0[i] : 0.0f
//
// src0: [N, 1, ne2, ne3]   (1D vector per batch)
// dst:  [N, N, ne2, ne3]   (diagonal matrix per batch)
//******************************************************************************

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <stdint.h>

સંરચના ggml_et_diag_params {
    સંરચના ggml_tensor src0;  // F32 input vector
    સંરચના ggml_tensor dst;   // F32 output diagonal matrix
};

પૂર્ણાંક entry_point(સંરચના ggml_et_diag_params * params, શૂન્ય * env) {
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

    જો (!src0_data || !dst_data) {
        પાછા -1;
    }

    અચલ int64_t ne0 = dst->ne[0];  // N (row width = column count)
    અચલ int64_t ne1 = dst->ne[1];  // N (number of rows)
    અચલ int64_t ne2 = dst->ne[2];
    અચલ int64_t ne3 = dst->ne[3];

    અચલ size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    અચલ size_t nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Total rows across all batches — parallelize over these
    અચલ int64_t total_rows = ne1 * ne2 * ne3;

    // Prepare zero vector for SIMD zeroing
    દશાંશ zero = 0.0f;
    __asm__ અસ્થિર("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

    માટે (int64_t row = thread_id; row < total_rows; row += અંક_threads) {
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);

        દશાંશ * dst_row = (દશાંશ *) ((અક્ષર *) dst_data + i1 * nb1 + i2 * nb2 + i3 * nb3);

        // Zero the entire row with SIMD
        int64_t       i0      = 0;
        અચલ int64_t vec_end = (ne0 / 8) * 8;
        માટે (; i0 < vec_end; i0 += 8) {
            __asm__ અસ્થિર("fsw.ps f10, %[d]\n" : [d] "=m"(*(દશાંશ (*)[8]) & dst_row[i0])::"f10");
        }
        માટે (; i0 < ne0; i0++) {
            dst_row[i0] = 0.0f;
        }

        // Place the diagonal element: dst[i1][i1] = src0[i1]
        અચલ દશાંશ * src_ptr = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i2 * nb02 + i3 * nb03);
        dst_row[i1]           = src_ptr[i1];
    }

    પાછા 0;
}
