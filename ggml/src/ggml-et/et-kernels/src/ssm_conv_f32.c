#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <stdint.h>

கட்டமைப்பு ggml_et_ssm_conv_params {
    கட்டமைப்பு ggml_tensor src0;  // conv_x: [d_conv - 1 + n_t, d_inner, n_seqs]
    கட்டமைப்பு ggml_tensor src1;  // conv1d.weight: [d_conv, d_inner]
    கட்டமைப்பு ggml_tensor dst;   // output: [d_inner, n_t, n_seqs]
};

முழு entry_point(கட்டமைப்பு ggml_et_ssm_conv_params * params, வெறுமை * env) {
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
    கட்டமைப்பு ggml_tensor * src1 = &params->src1;
    கட்டமைப்பு ggml_tensor * dst  = &params->dst;

    எனில் (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        திரும்பு -1;
    }

    மாறிலி தசம * src0_data = (மாறிலி தசம *) src0->data;
    மாறிலி தசம * src1_data = (மாறிலி தசம *) src1->data;
    தசம *       dst_data  = (தசம *) dst->data;

    எனில் (!src0_data || !src1_data || !dst_data) {
        திரும்பு -1;
    }

    மாறிலி int64_t nc  = src1->ne[0];
    மாறிலி int64_t ncs = src0->ne[0];
    மாறிலி int64_t nr  = src0->ne[1];
    மாறிலி int64_t n_t = dst->ne[1];
    மாறிலி int64_t n_s = dst->ne[2];

    எனில் (dst->ne[0] != nr || src1->ne[1] != nr || ncs != nc - 1 + n_t || src0->nb[0] != அளவு(தசம) ||
        src1->nb[0] != அளவு(தசம) || dst->nb[0] != அளவு(தசம) || src0->nb[1] != (size_t) ncs * அளவு(தசம) ||
        src1->nb[1] != (size_t) nc * அளவு(தசம)) {
        திரும்பு -1;
    }

    // Parallelize over d_inner in cache-line-aligned chunks (16 floats = 64B)
    மாறிலி int64_t chunk    = 16;
    மாறிலி int64_t n_chunks = (nr + chunk - 1) / chunk;

    // Save and set vector mask to all 8 lanes
    unsigned long saved_mask;
    __asm__ நிலையற்ற("mova.x.m %0" : "=r"(saved_mask));
    __asm__ நிலையற்ற("mov.m.x m0, x0, 0xFF");

    க்கு (int64_t i3 = 0; i3 < n_s; ++i3) {
        க்கு (int64_t i2 = 0; i2 < n_t; ++i2) {
            மாறிலி தசம * s = (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0_data + i2 * src0->nb[0] + i3 * src0->nb[2]);
            தசம *       x = (தசம *) ((எழுத்து *) dst_data + i2 * dst->nb[1] + i3 * dst->nb[2]);

            க்கு (int64_t ci = thread_id; ci < n_chunks; ci += இலக்கம்_threads) {
                மாறிலி int64_t i1_start = ci * chunk;
                மாறிலி int64_t i1_end   = i1_start + chunk < nr ? i1_start + chunk : nr;

                // Process 8 channels at a time with SIMD
                int64_t i1 = i1_start;
                க்கு (; i1 + 8 <= i1_end; i1 += 8) {
                    // Gather 8 channels' data into contiguous buffers for each tap
                    தசம tmp_s[8], tmp_c[8];
                    தசம acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

                    க்கு (int64_t i0 = 0; i0 < nc; ++i0) {
                        // TODO: Some way to get rid of this gather
                        க்கு (முழு j = 0; j < 8; ++j) {
                            tmp_s[j] = s[(i1 + j) * ncs + i0];
                            tmp_c[j] = src1_data[(i1 + j) * nc + i0];
                        }

                        __asm__ நிலையற்ற(
                            "flw.ps f10, %[acc]\n"
                            "flw.ps f11, %[sv]\n"
                            "flw.ps f12, %[cv]\n"
                            "fmadd.ps f10, f11, f12, f10\n"
                            "fsw.ps f10, %[out]\n"
                            : [out] "=m"(*(தசம (*)[8]) acc)
                            : [acc] "m"(*(மாறிலி தசம (*)[8]) acc), [sv] "m"(*(மாறிலி தசம (*)[8]) tmp_s),
                              [cv] "m"(*(மாறிலி தசம (*)[8]) tmp_c)
                            : "f10", "f11", "f12");
                    }

                    // Store 8 results — dst is contiguous along d_inner
                    __asm__ நிலையற்ற(
                        "flw.ps f10, %[acc]\n"
                        "fsw.ps f10, %[dst]\n"
                        : [dst] "=m"(*(தசம (*)[8])(x + i1))
                        : [acc] "m"(*(மாறிலி தசம (*)[8]) acc)
                        : "f10");
                }

                // Scalar tail for remaining channels
                க்கு (; i1 < i1_end; ++i1) {
                    மாறிலி தசம * c     = src1_data + i1 * nc;
                    மாறிலி தசம * s_row = s + i1 * ncs;
                    தசம         sumf  = 0.0f;
                    க்கு (int64_t i0 = 0; i0 < nc; ++i0) {
                        sumf += s_row[i0] * c[i0];
                    }
                    x[i1] = sumf;
                }
            }
        }
    }

    // Restore mask
    __asm__ நிலையற்ற("mova.m.x %0" ::"r"(saved_mask));

    திரும்பு 0;
}
