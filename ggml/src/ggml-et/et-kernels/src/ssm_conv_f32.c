#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <stdint.h>

સંરચના ggml_et_ssm_conv_params {
    સંરચના ggml_tensor src0;  // conv_x: [d_conv - 1 + n_t, d_inner, n_seqs]
    સંરચના ggml_tensor src1;  // conv1d.weight: [d_conv, d_inner]
    સંરચના ggml_tensor dst;   // output: [d_inner, n_t, n_seqs]
};

પૂર્ણાંક entry_point(સંરચના ggml_et_ssm_conv_params * params, શૂન્ય * env) {
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
    સંરચના ggml_tensor * src1 = &params->src1;
    સંરચના ggml_tensor * dst  = &params->dst;

    જો (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા -1;
    }

    અચલ દશાંશ * src0_data = (અચલ દશાંશ *) src0->data;
    અચલ દશાંશ * src1_data = (અચલ દશાંશ *) src1->data;
    દશાંશ *       dst_data  = (દશાંશ *) dst->data;

    જો (!src0_data || !src1_data || !dst_data) {
        પાછા -1;
    }

    અચલ int64_t nc  = src1->ne[0];
    અચલ int64_t ncs = src0->ne[0];
    અચલ int64_t nr  = src0->ne[1];
    અચલ int64_t n_t = dst->ne[1];
    અચલ int64_t n_s = dst->ne[2];

    જો (dst->ne[0] != nr || src1->ne[1] != nr || ncs != nc - 1 + n_t || src0->nb[0] != કદ(દશાંશ) ||
        src1->nb[0] != કદ(દશાંશ) || dst->nb[0] != કદ(દશાંશ) || src0->nb[1] != (size_t) ncs * કદ(દશાંશ) ||
        src1->nb[1] != (size_t) nc * કદ(દશાંશ)) {
        પાછા -1;
    }

    // Parallelize over d_inner in cache-line-aligned chunks (16 floats = 64B)
    અચલ int64_t chunk    = 16;
    અચલ int64_t n_chunks = (nr + chunk - 1) / chunk;

    // Save and set vector mask to all 8 lanes
    unsigned long saved_mask;
    __asm__ અસ્થિર("mova.x.m %0" : "=r"(saved_mask));
    __asm__ અસ્થિર("mov.m.x m0, x0, 0xFF");

    માટે (int64_t i3 = 0; i3 < n_s; ++i3) {
        માટે (int64_t i2 = 0; i2 < n_t; ++i2) {
            અચલ દશાંશ * s = (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + i2 * src0->nb[0] + i3 * src0->nb[2]);
            દશાંશ *       x = (દશાંશ *) ((અક્ષર *) dst_data + i2 * dst->nb[1] + i3 * dst->nb[2]);

            માટે (int64_t ci = thread_id; ci < n_chunks; ci += અંક_threads) {
                અચલ int64_t i1_start = ci * chunk;
                અચલ int64_t i1_end   = i1_start + chunk < nr ? i1_start + chunk : nr;

                // Process 8 channels at a time with SIMD
                int64_t i1 = i1_start;
                માટે (; i1 + 8 <= i1_end; i1 += 8) {
                    // Gather 8 channels' data into contiguous buffers for each tap
                    દશાંશ tmp_s[8], tmp_c[8];
                    દશાંશ acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

                    માટે (int64_t i0 = 0; i0 < nc; ++i0) {
                        // TODO: Some way to get rid of this gather
                        માટે (પૂર્ણાંક j = 0; j < 8; ++j) {
                            tmp_s[j] = s[(i1 + j) * ncs + i0];
                            tmp_c[j] = src1_data[(i1 + j) * nc + i0];
                        }

                        __asm__ અસ્થિર(
                            "flw.ps f10, %[acc]\n"
                            "flw.ps f11, %[sv]\n"
                            "flw.ps f12, %[cv]\n"
                            "fmadd.ps f10, f11, f12, f10\n"
                            "fsw.ps f10, %[out]\n"
                            : [out] "=m"(*(દશાંશ (*)[8]) acc)
                            : [acc] "m"(*(અચલ દશાંશ (*)[8]) acc), [sv] "m"(*(અચલ દશાંશ (*)[8]) tmp_s),
                              [cv] "m"(*(અચલ દશાંશ (*)[8]) tmp_c)
                            : "f10", "f11", "f12");
                    }

                    // Store 8 results — dst is contiguous along d_inner
                    __asm__ અસ્થિર(
                        "flw.ps f10, %[acc]\n"
                        "fsw.ps f10, %[dst]\n"
                        : [dst] "=m"(*(દશાંશ (*)[8])(x + i1))
                        : [acc] "m"(*(અચલ દશાંશ (*)[8]) acc)
                        : "f10");
                }

                // Scalar tail for remaining channels
                માટે (; i1 < i1_end; ++i1) {
                    અચલ દશાંશ * c     = src1_data + i1 * nc;
                    અચલ દશાંશ * s_row = s + i1 * ncs;
                    દશાંશ         sumf  = 0.0f;
                    માટે (int64_t i0 = 0; i0 < nc; ++i0) {
                        sumf += s_row[i0] * c[i0];
                    }
                    x[i1] = sumf;
                }
            }
        }
    }

    // Restore mask
    __asm__ અસ્થિર("mova.m.x %0" ::"r"(saved_mask));

    પાછા 0;
}
