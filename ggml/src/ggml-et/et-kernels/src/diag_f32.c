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

ساخت ggml_et_diag_params {
    ساخت ggml_tensor src0;  // F32 input vector
    ساخت ggml_tensor dst;   // F32 output diagonal matrix
};

صحیح اندراج_نقطہ(ساخت ggml_et_diag_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        واپس -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    اگر (thread_id < 0) {
        واپس 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        واپس -1;
    }

    ساخت ggml_tensor * src0 = &params->src0;
    ساخت ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس -1;
    }

    اعشاری * src0_ڈیٹا = (اعشاری *) src0->ڈیٹا;
    اعشاری * dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;

    اگر (!src0_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;
    }

    ثابت int64_t ne0 = dst->ne[0];  // N (row width = column count)
    ثابت int64_t ne1 = dst->ne[1];  // N (number of rows)
    ثابت int64_t ne2 = dst->ne[2];
    ثابت int64_t ne3 = dst->ne[3];

    ثابت size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ثابت size_t nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Total rows across all batches — parallelize over these
    ثابت int64_t کل_صفیں = ne1 * ne2 * ne3;

    // Prepare zero vector for SIMD zeroing
    اعشاری zero = 0.0f;
    __asm__ متلون("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

    برائے (int64_t صف = thread_id; صف < کل_صفیں; صف += عدد_دھاگے) {
        int64_t i1 = صف % ne1;
        int64_t i2 = (صف / ne1) % ne2;
        int64_t i3 = صف / (ne1 * ne2);

        اعشاری * dst_صف = (اعشاری *) ((حرف *) dst_ڈیٹا + i1 * nb1 + i2 * nb2 + i3 * nb3);

        // Zero the entire row with SIMD
        int64_t       i0      = 0;
        ثابت int64_t vec_انتہا = (ne0 / 8) * 8;
        برائے (; i0 < vec_انتہا; i0 += 8) {
            __asm__ متلون("fsw.ps f10, %[d]\n" : [d] "=m"(*(اعشاری (*)[8]) & dst_صف[i0])::"f10");
        }
        برائے (; i0 < ne0; i0++) {
            dst_صف[i0] = 0.0f;
        }

        // Place the diagonal element: dst[i1][i1] = src0[i1]
        ثابت اعشاری * src_ptr = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i2 * nb02 + i3 * nb03);
        dst_صف[i1]           = src_ptr[i1];
    }

    واپس 0;
}
