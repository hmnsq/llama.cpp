#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdint.h>

ساختار ggml_et_ssm_conv_params {
    ساختار ggml_tensor src0;  // conv_x: [d_conv - 1 + n_t, d_inner, n_seqs]
    ساختار ggml_tensor src1;  // conv1d.weight: [d_conv, d_inner]
    ساختار ggml_tensor dst;   // output: [d_inner, n_t, n_seqs]
};

صحیح entry_point(ساختار ggml_et_ssm_conv_params * params, تهی * env) {
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
    ساختار ggml_tensor * src1 = &params->src1;
    ساختار ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        بازگشت -1;
    }

    ثابت اعشاری * src0_data = (ثابت اعشاری *) src0->data;
    ثابت اعشاری * src1_data = (ثابت اعشاری *) src1->data;
    اعشاری *       dst_data  = (اعشاری *) dst->data;

    اگر (!src0_data || !src1_data || !dst_data) {
        بازگشت -1;
    }

    ثابت int64_t nc  = src1->ne[0];
    ثابت int64_t ncs = src0->ne[0];
    ثابت int64_t nr  = src0->ne[1];
    ثابت int64_t n_t = dst->ne[1];
    ثابت int64_t n_s = dst->ne[2];

    اگر (dst->ne[0] != nr || src1->ne[1] != nr || ncs != nc - 1 + n_t || src0->nb[0] != اندازه(اعشاری) ||
        src1->nb[0] != اندازه(اعشاری) || dst->nb[0] != اندازه(اعشاری) || src0->nb[1] != (size_t) ncs * اندازه(اعشاری) ||
        src1->nb[1] != (size_t) nc * اندازه(اعشاری)) {
        بازگشت -1;
    }

    // Parallelize over d_inner in cache-line-aligned chunks (16 floats = 64B)
    ثابت int64_t chunk    = 16;
    ثابت int64_t n_chunks = (nr + chunk - 1) / chunk;

    // Save and set vector mask to all 8 lanes
    unsigned long saved_mask;
    __asm__ فرار("mova.x.m %0" : "=r"(saved_mask));
    __asm__ فرار("mov.m.x m0, x0, 0xFF");

    برای (int64_t i3 = 0; i3 < n_s; ++i3) {
        برای (int64_t i2 = 0; i2 < n_t; ++i2) {
            ثابت اعشاری * s = (ثابت اعشاری *) ((ثابت نویسه *) src0_data + i2 * src0->nb[0] + i3 * src0->nb[2]);
            اعشاری *       x = (اعشاری *) ((نویسه *) dst_data + i2 * dst->nb[1] + i3 * dst->nb[2]);

            برای (int64_t ci = thread_id; ci < n_chunks; ci += رقم_threads) {
                ثابت int64_t i1_start = ci * chunk;
                ثابت int64_t i1_end   = i1_start + chunk < nr ? i1_start + chunk : nr;

                // Process 8 channels at a time with SIMD
                int64_t i1 = i1_start;
                برای (; i1 + 8 <= i1_end; i1 += 8) {
                    // Gather 8 channels' data into contiguous buffers for each tap
                    اعشاری tmp_s[8], tmp_c[8];
                    اعشاری acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

                    برای (int64_t i0 = 0; i0 < nc; ++i0) {
                        // TODO: Some way to get rid of this gather
                        برای (صحیح j = 0; j < 8; ++j) {
                            tmp_s[j] = s[(i1 + j) * ncs + i0];
                            tmp_c[j] = src1_data[(i1 + j) * nc + i0];
                        }

                        __asm__ فرار(
                            "flw.ps f10, %[acc]\n"
                            "flw.ps f11, %[sv]\n"
                            "flw.ps f12, %[cv]\n"
                            "fmadd.ps f10, f11, f12, f10\n"
                            "fsw.ps f10, %[out]\n"
                            : [out] "=m"(*(اعشاری (*)[8]) acc)
                            : [acc] "m"(*(ثابت اعشاری (*)[8]) acc), [sv] "m"(*(ثابت اعشاری (*)[8]) tmp_s),
                              [cv] "m"(*(ثابت اعشاری (*)[8]) tmp_c)
                            : "f10", "f11", "f12");
                    }

                    // Store 8 results — dst is contiguous along d_inner
                    __asm__ فرار(
                        "flw.ps f10, %[acc]\n"
                        "fsw.ps f10, %[dst]\n"
                        : [dst] "=m"(*(اعشاری (*)[8])(x + i1))
                        : [acc] "m"(*(ثابت اعشاری (*)[8]) acc)
                        : "f10");
                }

                // Scalar tail for remaining channels
                برای (; i1 < i1_end; ++i1) {
                    ثابت اعشاری * c     = src1_data + i1 * nc;
                    ثابت اعشاری * s_row = s + i1 * ncs;
                    اعشاری         sumf  = 0.0f;
                    برای (int64_t i0 = 0; i0 < nc; ++i0) {
                        sumf += s_row[i0] * c[i0];
                    }
                    x[i1] = sumf;
                }
            }
        }
    }

    // Restore mask
    __asm__ فرار("mova.m.x %0" ::"r"(saved_mask));

    بازگشت 0;
}
