#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_ssm_conv_params {
    بنية ggml_tensor src0;  // conv_x: [d_conv - 1 + n_t, d_inner, n_seqs]
    بنية ggml_tensor src1;  // conv1d.weight: [d_conv, d_inner]
    بنية ggml_tensor dst;   // output: [d_inner, n_t, n_seqs]
};

صحيح entry_point(بنية ggml_et_ssm_conv_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    صحيح عدد_threads = get_عدد_threads(kernel_env->shire_mask);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * src1 = &params->src1;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    ثابت عائم * src0_data = (ثابت عائم *) src0->data;
    ثابت عائم * src1_data = (ثابت عائم *) src1->data;
    عائم *       dst_data  = (عائم *) dst->data;

    إذا (!src0_data || !src1_data || !dst_data) {
        أرجع -1;
    }

    ثابت int64_t nc  = src1->ne[0];
    ثابت int64_t ncs = src0->ne[0];
    ثابت int64_t nr  = src0->ne[1];
    ثابت int64_t n_t = dst->ne[1];
    ثابت int64_t n_s = dst->ne[2];

    إذا (dst->ne[0] != nr || src1->ne[1] != nr || ncs != nc - 1 + n_t || src0->nb[0] != حجم(عائم) ||
        src1->nb[0] != حجم(عائم) || dst->nb[0] != حجم(عائم) || src0->nb[1] != (size_t) ncs * حجم(عائم) ||
        src1->nb[1] != (size_t) nc * حجم(عائم)) {
        أرجع -1;
    }

    // Parallelize over d_inner in cache-line-aligned chunks (16 floats = 64B)
    ثابت int64_t chunk    = 16;
    ثابت int64_t n_chunks = (nr + chunk - 1) / chunk;

    // Save and set vector mask to all 8 lanes
    unsigned long saved_mask;
    __asm__ متطاير("mova.x.m %0" : "=r"(saved_mask));
    __asm__ متطاير("mov.m.x m0, x0, 0xFF");

    لكل (int64_t i3 = 0; i3 < n_s; ++i3) {
        لكل (int64_t i2 = 0; i2 < n_t; ++i2) {
            ثابت عائم * s = (ثابت عائم *) ((ثابت حرف *) src0_data + i2 * src0->nb[0] + i3 * src0->nb[2]);
            عائم *       x = (عائم *) ((حرف *) dst_data + i2 * dst->nb[1] + i3 * dst->nb[2]);

            لكل (int64_t ci = thread_id; ci < n_chunks; ci += عدد_threads) {
                ثابت int64_t i1_start = ci * chunk;
                ثابت int64_t i1_end   = i1_start + chunk < nr ? i1_start + chunk : nr;

                // Process 8 channels at a time with SIMD
                int64_t i1 = i1_start;
                لكل (; i1 + 8 <= i1_end; i1 += 8) {
                    // Gather 8 channels' data into contiguous buffers for each tap
                    عائم tmp_s[8], tmp_c[8];
                    عائم acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

                    لكل (int64_t i0 = 0; i0 < nc; ++i0) {
                        // TODO: Some way to get rid of this gather
                        لكل (صحيح j = 0; j < 8; ++j) {
                            tmp_s[j] = s[(i1 + j) * ncs + i0];
                            tmp_c[j] = src1_data[(i1 + j) * nc + i0];
                        }

                        __asm__ متطاير(
                            "flw.ps f10, %[acc]\n"
                            "flw.ps f11, %[sv]\n"
                            "flw.ps f12, %[cv]\n"
                            "fmadd.ps f10, f11, f12, f10\n"
                            "fsw.ps f10, %[out]\n"
                            : [out] "=m"(*(عائم (*)[8]) acc)
                            : [acc] "m"(*(ثابت عائم (*)[8]) acc), [sv] "m"(*(ثابت عائم (*)[8]) tmp_s),
                              [cv] "m"(*(ثابت عائم (*)[8]) tmp_c)
                            : "f10", "f11", "f12");
                    }

                    // Store 8 results — dst is contiguous along d_inner
                    __asm__ متطاير(
                        "flw.ps f10, %[acc]\n"
                        "fsw.ps f10, %[dst]\n"
                        : [dst] "=m"(*(عائم (*)[8])(x + i1))
                        : [acc] "m"(*(ثابت عائم (*)[8]) acc)
                        : "f10");
                }

                // Scalar tail for remaining channels
                لكل (; i1 < i1_end; ++i1) {
                    ثابت عائم * c     = src1_data + i1 * nc;
                    ثابت عائم * s_row = s + i1 * ncs;
                    عائم         sumf  = 0.0f;
                    لكل (int64_t i0 = 0; i0 < nc; ++i0) {
                        sumf += s_row[i0] * c[i0];
                    }
                    x[i1] = sumf;
                }
            }
        }
    }

    // Restore mask
    __asm__ متطاير("mova.m.x %0" ::"r"(saved_mask));

    أرجع 0;
}
