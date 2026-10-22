//******************************************************************************
// SUM_ROWS F32 Kernel
// Row-wise sum reduction: dst[0, i1, i2, i3] = sum(src0[0..ne00-1, i1, i2, i3])
// Vectorized 8-wide accumulation with horizontal reduction.
//******************************************************************************

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_ਜੋੜ_ਪੰਕਤੀਆਂ_params {
    ਬਣਤਰ ggml_tensor src0;  // F32 input tensor [ne00, ne01, ne02, ne03]
    ਬਣਤਰ ggml_tensor dst;   // F32 output tensor [1, ne01, ne02, ne03]
};

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_ਜੋੜ_ਪੰਕਤੀਆਂ_params * params, ਖਾਲੀ * env) {
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


    ਅਚਲ int64_t ne00 = src0->ne[0];  // Row length (to be summed)
    ਅਚਲ int64_t ne01 = src0->ne[1];
    ਅਚਲ int64_t ne02 = src0->ne[2];
    ਅਚਲ int64_t ne03 = src0->ne[3];

    ਅਚਲ size_t nb01 = src0->nb[1];
    ਅਚਲ size_t nb02 = src0->nb[2];
    ਅਚਲ size_t nb03 = src0->nb[3];

    ਅਚਲ size_t nb1 = dst->nb[1];
    ਅਚਲ size_t nb2 = dst->nb[2];
    ਅਚਲ size_t nb3 = dst->nb[3];

    // Flatten rows across dimensions 1,2,3 and distribute across threads
    ਅਚਲ int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = ne01 * ne02 * ne03;

    ਲਈ (int64_t ir = thread_id; ir < ਕੁੱਲ_ਪੰਕਤੀਆਂ; ir += ਅੰਕ_ਧਾਗੇ) {
        ਅਚਲ int64_t i03 = ir / (ne02 * ne01);
        ਅਚਲ int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        ਅਚਲ int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

        ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i01 * nb01 + i02 * nb02 + i03 * nb03);
        ਦਸ਼ਮਲਵ *       dst_ptr = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + i01 * nb1 + i02 * nb2 + i03 * nb3);

        // Vectorized 8-wide sum accumulation
        ਦਸ਼ਮਲਵ zero = 0.0f;
        __asm__ ਅਸਥਿਰ("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        ਲਈ (int32_t i0 = 0; i0 < (int32_t) ne00; i0 += 8) {
            __asm__ ਅਸਥਿਰ(
                "flw.ps f11, %[x_vec]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & src_ਪੰਕਤੀ[i0])
                : "f10", "f11");
        }

        // Horizontal sum of 8 accumulated values in f10
        ਦਸ਼ਮਲਵ ਪੰਕਤੀ_ਜੋੜ;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(ਪੰਕਤੀ_ਜੋੜ)::"t0", "f1", "f2", "f3", "f4", "f5");

        atomic_store_f32(dst_ptr, ਪੰਕਤੀ_ਜੋੜ);
    }

    ਵਾਪਸ 0;
}
