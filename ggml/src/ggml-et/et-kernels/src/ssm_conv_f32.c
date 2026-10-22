#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_ssm_conv_params {
    ਬਣਤਰ ggml_tensor src0;  // conv_x: [d_conv - 1 + n_t, d_inner, n_seqs]
    ਬਣਤਰ ggml_tensor src1;  // conv1d.weight: [d_conv, d_inner]
    ਬਣਤਰ ggml_tensor dst;   // output: [d_inner, n_t, n_seqs]
};

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_ssm_conv_params * params, ਖਾਲੀ * env) {
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
    ਬਣਤਰ ggml_tensor * src1 = &params->src1;
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;

    ਜੇ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ ਦਸ਼ਮਲਵ * src0_ਅੰਕੜੇ = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * src1_ਅੰਕੜੇ = (ਅਚਲ ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ *       dst_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਜੇ (!src0_ਅੰਕੜੇ || !src1_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t nc  = src1->ne[0];
    ਅਚਲ int64_t ncs = src0->ne[0];
    ਅਚਲ int64_t nr  = src0->ne[1];
    ਅਚਲ int64_t n_t = dst->ne[1];
    ਅਚਲ int64_t n_s = dst->ne[2];

    ਜੇ (dst->ne[0] != nr || src1->ne[1] != nr || ncs != nc - 1 + n_t || src0->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) ||
        src1->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || dst->nb[0] != ਆਕਾਰ(ਦਸ਼ਮਲਵ) || src0->nb[1] != (size_t) ncs * ਆਕਾਰ(ਦਸ਼ਮਲਵ) ||
        src1->nb[1] != (size_t) nc * ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
        ਵਾਪਸ -1;
    }

    // Parallelize over d_inner in cache-line-aligned chunks (16 floats = 64B)
    ਅਚਲ int64_t chunk    = 16;
    ਅਚਲ int64_t n_chunks = (nr + chunk - 1) / chunk;

    // Save and set vector mask to all 8 lanes
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ saved_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(saved_ਨਕਾਬ));
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");

    ਲਈ (int64_t i3 = 0; i3 < n_s; ++i3) {
        ਲਈ (int64_t i2 = 0; i2 < n_t; ++i2) {
            ਅਚਲ ਦਸ਼ਮਲਵ * s = (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + i2 * src0->nb[0] + i3 * src0->nb[2]);
            ਦਸ਼ਮਲਵ *       x = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + i2 * dst->nb[1] + i3 * dst->nb[2]);

            ਲਈ (int64_t ci = thread_id; ci < n_chunks; ci += ਅੰਕ_ਧਾਗੇ) {
                ਅਚਲ int64_t i1_ਸ਼ੁਰੂ = ci * chunk;
                ਅਚਲ int64_t i1_ਸਿਰਾ   = i1_ਸ਼ੁਰੂ + chunk < nr ? i1_ਸ਼ੁਰੂ + chunk : nr;

                // Process 8 channels at a time with SIMD
                int64_t i1 = i1_ਸ਼ੁਰੂ;
                ਲਈ (; i1 + 8 <= i1_ਸਿਰਾ; i1 += 8) {
                    // Gather 8 channels' data into contiguous buffers for each tap
                    ਦਸ਼ਮਲਵ tmp_s[8], tmp_c[8];
                    ਦਸ਼ਮਲਵ acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

                    ਲਈ (int64_t i0 = 0; i0 < nc; ++i0) {
                        // TODO: Some way to get rid of this gather
                        ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
                            tmp_s[j] = s[(i1 + j) * ncs + i0];
                            tmp_c[j] = src1_ਅੰਕੜੇ[(i1 + j) * nc + i0];
                        }

                        __asm__ ਅਸਥਿਰ(
                            "flw.ps f10, %[acc]\n"
                            "flw.ps f11, %[sv]\n"
                            "flw.ps f12, %[cv]\n"
                            "fmadd.ps f10, f11, f12, f10\n"
                            "fsw.ps f10, %[out]\n"
                            : [ਬਾਹਰਲਾ] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) acc)
                            : [acc] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) acc), [sv] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) tmp_s),
                              [cv] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) tmp_c)
                            : "f10", "f11", "f12");
                    }

                    // Store 8 results — dst is contiguous along d_inner
                    __asm__ ਅਸਥਿਰ(
                        "flw.ps f10, %[acc]\n"
                        "fsw.ps f10, %[dst]\n"
                        : [dst] "=m"(*(ਦਸ਼ਮਲਵ (*)[8])(x + i1))
                        : [acc] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) acc)
                        : "f10");
                }

                // Scalar tail for remaining channels
                ਲਈ (; i1 < i1_ਸਿਰਾ; ++i1) {
                    ਅਚਲ ਦਸ਼ਮਲਵ * c     = src1_ਅੰਕੜੇ + i1 * nc;
                    ਅਚਲ ਦਸ਼ਮਲਵ * s_ਪੰਕਤੀ = s + i1 * ncs;
                    ਦਸ਼ਮਲਵ         sumf  = 0.0f;
                    ਲਈ (int64_t i0 = 0; i0 < nc; ++i0) {
                        sumf += s_ਪੰਕਤੀ[i0] * c[i0];
                    }
                    x[i1] = sumf;
                }
            }
        }
    }

    // Restore mask
    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(saved_ਨਕਾਬ));

    ਵਾਪਸ 0;
}
