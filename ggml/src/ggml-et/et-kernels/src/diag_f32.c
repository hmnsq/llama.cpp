//******************************************************************************
// Diag F32 Kernel
// Creates a diagonal matrix from a 1D vector.
// dst[i][j] = (i == j) ? src0[i] : 0.0f
//
// src0: [N, 1, ne2, ne3]   (1D vector per batch)
// dst:  [N, N, ne2, ne3]   (diagonal matrix per batch)
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_diag_params {
    بنية ggml_tensor src0;  // F32 input vector
    بنية ggml_tensor dst;   // F32 output diagonal matrix
};

صحيح اندراج_نقطہ(بنية ggml_et_diag_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    عائم * src0_انکڑے = (عائم *) src0->انکڑے;
    عائم * dst_انکڑے  = (عائم *) dst->انکڑے;

    إذا (!src0_انکڑے || !dst_انکڑے) {
        أرجع -1;
    }

    ثابت int64_t ne0 = dst->ne[0];  // N (row width = column count)
    ثابت int64_t ne1 = dst->ne[1];  // N (number of rows)
    ثابت int64_t ne2 = dst->ne[2];
    ثابت int64_t ne3 = dst->ne[3];

    ثابت size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ثابت size_t nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Total rows across all batches — parallelize over these
    ثابت int64_t کل_صفاں = ne1 * ne2 * ne3;

    // Prepare zero vector for SIMD zeroing
    عائم zero = 0.0f;
    __asm__ متطاير("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

    لكل (int64_t صف = thread_id; صف < کل_صفاں; صف += عدد_دھاگے) {
        int64_t i1 = صف % ne1;
        int64_t i2 = (صف / ne1) % ne2;
        int64_t i3 = صف / (ne1 * ne2);

        عائم * dst_صف = (عائم *) ((حرف *) dst_انکڑے + i1 * nb1 + i2 * nb2 + i3 * nb3);

        // Zero the entire row with SIMD
        int64_t       i0      = 0;
        ثابت int64_t vec_سرا = (ne0 / 8) * 8;
        لكل (; i0 < vec_سرا; i0 += 8) {
            __asm__ متطاير("fsw.ps f10, %[d]\n" : [d] "=m"(*(عائم (*)[8]) & dst_صف[i0])::"f10");
        }
        لكل (; i0 < ne0; i0++) {
            dst_صف[i0] = 0.0f;
        }

        // Place the diagonal element: dst[i1][i1] = src0[i1]
        ثابت عائم * src_ptr = (ثابت عائم *) ((ثابت حرف *) src0_انکڑے + i2 * nb02 + i3 * nb03);
        dst_صف[i1]           = src_ptr[i1];
    }

    أرجع 0;
}
