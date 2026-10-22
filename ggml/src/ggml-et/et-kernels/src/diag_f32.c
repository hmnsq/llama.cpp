//******************************************************************************
// Diag F32 Kernel
// Creates a diagonal matrix from a 1D vector.
// dst[i][j] = (i == j) ? src0[i] : 0.0f
//
// src0: [N, 1, ne2, ne3]   (1D vector per batch)
// dst:  [N, N, ne2, ne3]   (diagonal matrix per batch)
//******************************************************************************

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_diag_params {
    ਬਣਤਰ ggml_tensor src0;  // F32 input vector
    ਬਣਤਰ ggml_tensor dst;   // F32 output diagonal matrix
};

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_diag_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ਜੇ (!kernel_env) {
        ਵਾਪਸ -1;
    }

    ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);

    ਜੇ (thread_id < 0) {
        ਵਾਪਸ 0;
    }

    ਜੇ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;
    }

    ਬਣਤਰ ggml_tensor * src0 = &params->src0;
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;

    ਜੇ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;
    }

    ਦਸ਼ਮਲਵ * src0_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਜੇ (!src0_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t ne0 = dst->ne[0];  // N (row width = column count)
    ਅਚਲ int64_t ne1 = dst->ne[1];  // N (number of rows)
    ਅਚਲ int64_t ne2 = dst->ne[2];
    ਅਚਲ int64_t ne3 = dst->ne[3];

    ਅਚਲ size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    ਅਚਲ size_t nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Total rows across all batches — parallelize over these
    ਅਚਲ int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = ne1 * ne2 * ne3;

    // Prepare zero vector for SIMD zeroing
    ਦਸ਼ਮਲਵ zero = 0.0f;
    __asm__ ਅਸਥਿਰ("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

    ਲਈ (int64_t ਪੰਕਤੀ = thread_id; ਪੰਕਤੀ < ਕੁੱਲ_ਪੰਕਤੀਆਂ; ਪੰਕਤੀ += ਅੰਕ_ਧਾਗੇ) {
        int64_t i1 = ਪੰਕਤੀ % ne1;
        int64_t i2 = (ਪੰਕਤੀ / ne1) % ne2;
        int64_t i3 = ਪੰਕਤੀ / (ne1 * ne2);

        ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + i1 * nb1 + i2 * nb2 + i3 * nb3);

        // Zero the entire row with SIMD
        int64_t       i0      = 0;
        ਅਚਲ int64_t vec_ਸਿਰਾ = (ne0 / 8) * 8;
        ਲਈ (; i0 < vec_ਸਿਰਾ; i0 += 8) {
            __asm__ ਅਸਥਿਰ("fsw.ps f10, %[d]\n" : [d] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ਪੰਕਤੀ[i0])::"f10");
        }
        ਲਈ (; i0 < ne0; i0++) {
            dst_ਪੰਕਤੀ[i0] = 0.0f;
        }

        // Place the diagonal element: dst[i1][i1] = src0[i1]
        ਅਚਲ ਦਸ਼ਮਲਵ * src_ptr = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i2 * nb02 + i3 * nb03);
        dst_ਪੰਕਤੀ[i1]           = src_ptr[i1];
    }

    ਵਾਪਸ 0;
}
