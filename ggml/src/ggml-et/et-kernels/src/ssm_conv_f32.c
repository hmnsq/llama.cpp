#شامل "ggml_tensor.h"
#شامل "platform.h"

#شامل <stdint.h>

ساخت ggml_et_ssm_conv_params {
    ساخت ggml_tensor src0;  // conv_x: [d_conv - 1 + n_t, d_inner, n_seqs]
    ساخت ggml_tensor src1;  // conv1d.weight: [d_conv, d_inner]
    ساخت ggml_tensor dst;   // output: [d_inner, n_t, n_seqs]
};

صحیح اندراج_نقطہ(ساخت ggml_et_ssm_conv_params * params, خالی * env) {
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
    ساخت ggml_tensor * src1 = &params->src1;
    ساخت ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس -1;
    }

    ثابت اعشاری * src0_ڈیٹا = (ثابت اعشاری *) src0->ڈیٹا;
    ثابت اعشاری * src1_ڈیٹا = (ثابت اعشاری *) src1->ڈیٹا;
    اعشاری *       dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;

    اگر (!src0_ڈیٹا || !src1_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;
    }

    ثابت int64_t nc  = src1->ne[0];
    ثابت int64_t ncs = src0->ne[0];
    ثابت int64_t nr  = src0->ne[1];
    ثابت int64_t n_t = dst->ne[1];
    ثابت int64_t n_s = dst->ne[2];

    اگر (dst->ne[0] != nr || src1->ne[1] != nr || ncs != nc - 1 + n_t || src0->nb[0] != حجم(اعشاری) ||
        src1->nb[0] != حجم(اعشاری) || dst->nb[0] != حجم(اعشاری) || src0->nb[1] != (size_t) ncs * حجم(اعشاری) ||
        src1->nb[1] != (size_t) nc * حجم(اعشاری)) {
        واپس -1;
    }

    // Parallelize over d_inner in cache-line-aligned chunks (16 floats = 64B)
    ثابت int64_t chunk    = 16;
    ثابت int64_t n_chunks = (nr + chunk - 1) / chunk;

    // Save and set vector mask to all 8 lanes
    غیرعلامتی طویل saved_نقاب;
    __asm__ متلون("mova.x.m %0" : "=r"(saved_نقاب));
    __asm__ متلون("mov.m.x m0, x0, 0xFF");

    برائے (int64_t i3 = 0; i3 < n_s; ++i3) {
        برائے (int64_t i2 = 0; i2 < n_t; ++i2) {
            ثابت اعشاری * s = (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + i2 * src0->nb[0] + i3 * src0->nb[2]);
            اعشاری *       x = (اعشاری *) ((حرف *) dst_ڈیٹا + i2 * dst->nb[1] + i3 * dst->nb[2]);

            برائے (int64_t ci = thread_id; ci < n_chunks; ci += عدد_دھاگے) {
                ثابت int64_t i1_آغاز = ci * chunk;
                ثابت int64_t i1_انتہا   = i1_آغاز + chunk < nr ? i1_آغاز + chunk : nr;

                // Process 8 channels at a time with SIMD
                int64_t i1 = i1_آغاز;
                برائے (; i1 + 8 <= i1_انتہا; i1 += 8) {
                    // Gather 8 channels' data into contiguous buffers for each tap
                    اعشاری tmp_s[8], tmp_c[8];
                    اعشاری acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

                    برائے (int64_t i0 = 0; i0 < nc; ++i0) {
                        // TODO: Some way to get rid of this gather
                        برائے (صحیح j = 0; j < 8; ++j) {
                            tmp_s[j] = s[(i1 + j) * ncs + i0];
                            tmp_c[j] = src1_ڈیٹا[(i1 + j) * nc + i0];
                        }

                        __asm__ متلون(
                            "flw.ps f10, %[acc]\n"
                            "flw.ps f11, %[sv]\n"
                            "flw.ps f12, %[cv]\n"
                            "fmadd.ps f10, f11, f12, f10\n"
                            "fsw.ps f10, %[out]\n"
                            : [برآمد] "=m"(*(اعشاری (*)[8]) acc)
                            : [acc] "m"(*(ثابت اعشاری (*)[8]) acc), [sv] "m"(*(ثابت اعشاری (*)[8]) tmp_s),
                              [cv] "m"(*(ثابت اعشاری (*)[8]) tmp_c)
                            : "f10", "f11", "f12");
                    }

                    // Store 8 results — dst is contiguous along d_inner
                    __asm__ متلون(
                        "flw.ps f10, %[acc]\n"
                        "fsw.ps f10, %[dst]\n"
                        : [dst] "=m"(*(اعشاری (*)[8])(x + i1))
                        : [acc] "m"(*(ثابت اعشاری (*)[8]) acc)
                        : "f10");
                }

                // Scalar tail for remaining channels
                برائے (; i1 < i1_انتہا; ++i1) {
                    ثابت اعشاری * c     = src1_ڈیٹا + i1 * nc;
                    ثابت اعشاری * s_صف = s + i1 * ncs;
                    اعشاری         sumf  = 0.0f;
                    برائے (int64_t i0 = 0; i0 < nc; ++i0) {
                        sumf += s_صف[i0] * c[i0];
                    }
                    x[i1] = sumf;
                }
            }
        }
    }

    // Restore mask
    __asm__ متلون("mova.m.x %0" ::"r"(saved_نقاب));

    واپس 0;
}
