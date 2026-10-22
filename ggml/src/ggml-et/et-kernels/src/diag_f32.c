//******************************************************************************
// Diag F32 Kernel
// Creates a diagonal matrix from a 1D vector.
// dst[i][j] = (i == j) ? src0[i] : 0.0f
//
// src0: [N, 1, ne2, ne3]   (1D vector per batch)
// dst:  [N, N, ne2, ne3]   (diagonal matrix per batch)
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdint.h>

ساختار ggml_et_diag_params {
    ساختار ggml_tensor src0;  // F32 input vector
    ساختار ggml_tensor dst;   // F32 output diagonal matrix
};

صحیح entry_point(ساختار ggml_et_diag_params * params, تهی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        بازگشت -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحیح رقم_threads = get_رقم_threads(kernel_env->shire_mask);

    اگر (thread_id < 0) {
        بازگشت 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        بازگشت -1;
    }

    ساختار ggml_tensor * src0 = &params->src0;
    ساختار ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        بازگشت -1;
    }

    اعشاری * src0_data = (اعشاری *) src0->data;
    اعشاری * dst_data  = (اعشاری *) dst->data;

    اگر (!src0_data || !dst_data) {
        بازگشت -1;
    }

    ثابت int64_t ne0 = dst->ne[0];  // N (row width = column count)
    ثابت int64_t ne1 = dst->ne[1];  // N (number of rows)
    ثابت int64_t ne2 = dst->ne[2];
    ثابت int64_t ne3 = dst->ne[3];

    ثابت size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ثابت size_t nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Total rows across all batches — parallelize over these
    ثابت int64_t total_rows = ne1 * ne2 * ne3;

    // Prepare zero vector for SIMD zeroing
    اعشاری zero = 0.0f;
    __asm__ فرار("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

    برای (int64_t row = thread_id; row < total_rows; row += رقم_threads) {
        int64_t i1 = row % ne1;
        int64_t i2 = (row / ne1) % ne2;
        int64_t i3 = row / (ne1 * ne2);

        اعشاری * dst_row = (اعشاری *) ((نویسه *) dst_data + i1 * nb1 + i2 * nb2 + i3 * nb3);

        // Zero the entire row with SIMD
        int64_t       i0      = 0;
        ثابت int64_t vec_end = (ne0 / 8) * 8;
        برای (; i0 < vec_end; i0 += 8) {
            __asm__ فرار("fsw.ps f10, %[d]\n" : [d] "=m"(*(اعشاری (*)[8]) & dst_row[i0])::"f10");
        }
        برای (; i0 < ne0; i0++) {
            dst_row[i0] = 0.0f;
        }

        // Place the diagonal element: dst[i1][i1] = src0[i1]
        ثابت اعشاری * src_ptr = (ثابت اعشاری *) ((ثابت نویسه *) src0_data + i2 * nb02 + i3 * nb03);
        dst_row[i1]           = src_ptr[i1];
    }

    بازگشت 0;
}
