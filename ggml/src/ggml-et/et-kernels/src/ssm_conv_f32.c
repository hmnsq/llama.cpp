#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <stdint.h>

কাঠামো ggml_et_ssm_conv_params {
    কাঠামো ggml_tensor src0;  // conv_x: [d_conv - 1 + n_t, d_inner, n_seqs]
    কাঠামো ggml_tensor src1;  // conv1d.weight: [d_conv, d_inner]
    কাঠামো ggml_tensor dst;   // output: [d_inner, n_t, n_seqs]
};

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_ssm_conv_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ফেরত -1;
    }

    পূর্ণসংখ্যা thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণসংখ্যা অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ফেরত 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ফেরত -1;
    }

    কাঠামো ggml_tensor * src0 = &params->src0;
    কাঠামো ggml_tensor * src1 = &params->src1;
    কাঠামো ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ফেরত -1;
    }

    ধ্রুবক দশমিক * src0_তথ্য = (ধ্রুবক দশমিক *) src0->তথ্য;
    ধ্রুবক দশমিক * src1_তথ্য = (ধ্রুবক দশমিক *) src1->তথ্য;
    দশমিক *       dst_তথ্য  = (দশমিক *) dst->তথ্য;

    যদি (!src0_তথ্য || !src1_তথ্য || !dst_তথ্য) {
        ফেরত -1;
    }

    ধ্রুবক int64_t nc  = src1->ne[0];
    ধ্রুবক int64_t ncs = src0->ne[0];
    ধ্রুবক int64_t nr  = src0->ne[1];
    ধ্রুবক int64_t n_t = dst->ne[1];
    ধ্রুবক int64_t n_s = dst->ne[2];

    যদি (dst->ne[0] != nr || src1->ne[1] != nr || ncs != nc - 1 + n_t || src0->nb[0] != আকার(দশমিক) ||
        src1->nb[0] != আকার(দশমিক) || dst->nb[0] != আকার(দশমিক) || src0->nb[1] != (size_t) ncs * আকার(দশমিক) ||
        src1->nb[1] != (size_t) nc * আকার(দশমিক)) {
        ফেরত -1;
    }

    // Parallelize over d_inner in cache-line-aligned chunks (16 floats = 64B)
    ধ্রুবক int64_t chunk    = 16;
    ধ্রুবক int64_t n_chunks = (nr + chunk - 1) / chunk;

    // Save and set vector mask to all 8 lanes
    unsigned long saved_mask;
    __asm__ অস্থির("mova.x.m %0" : "=r"(saved_mask));
    __asm__ অস্থির("mov.m.x m0, x0, 0xFF");

    জন্য (int64_t i3 = 0; i3 < n_s; ++i3) {
        জন্য (int64_t i2 = 0; i2 < n_t; ++i2) {
            ধ্রুবক দশমিক * s = (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src0_তথ্য + i2 * src0->nb[0] + i3 * src0->nb[2]);
            দশমিক *       x = (দশমিক *) ((অক্ষর *) dst_তথ্য + i2 * dst->nb[1] + i3 * dst->nb[2]);

            জন্য (int64_t ci = thread_id; ci < n_chunks; ci += অঙ্ক_threads) {
                ধ্রুবক int64_t i1_start = ci * chunk;
                ধ্রুবক int64_t i1_end   = i1_start + chunk < nr ? i1_start + chunk : nr;

                // Process 8 channels at a time with SIMD
                int64_t i1 = i1_start;
                জন্য (; i1 + 8 <= i1_end; i1 += 8) {
                    // Gather 8 channels' data into contiguous buffers for each tap
                    দশমিক tmp_s[8], tmp_c[8];
                    দশমিক acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

                    জন্য (int64_t i0 = 0; i0 < nc; ++i0) {
                        // TODO: Some way to get rid of this gather
                        জন্য (পূর্ণসংখ্যা j = 0; j < 8; ++j) {
                            tmp_s[j] = s[(i1 + j) * ncs + i0];
                            tmp_c[j] = src1_তথ্য[(i1 + j) * nc + i0];
                        }

                        __asm__ অস্থির(
                            "flw.ps f10, %[acc]\n"
                            "flw.ps f11, %[sv]\n"
                            "flw.ps f12, %[cv]\n"
                            "fmadd.ps f10, f11, f12, f10\n"
                            "fsw.ps f10, %[out]\n"
                            : [out] "=m"(*(দশমিক (*)[8]) acc)
                            : [acc] "m"(*(ধ্রুবক দশমিক (*)[8]) acc), [sv] "m"(*(ধ্রুবক দশমিক (*)[8]) tmp_s),
                              [cv] "m"(*(ধ্রুবক দশমিক (*)[8]) tmp_c)
                            : "f10", "f11", "f12");
                    }

                    // Store 8 results — dst is contiguous along d_inner
                    __asm__ অস্থির(
                        "flw.ps f10, %[acc]\n"
                        "fsw.ps f10, %[dst]\n"
                        : [dst] "=m"(*(দশমিক (*)[8])(x + i1))
                        : [acc] "m"(*(ধ্রুবক দশমিক (*)[8]) acc)
                        : "f10");
                }

                // Scalar tail for remaining channels
                জন্য (; i1 < i1_end; ++i1) {
                    ধ্রুবক দশমিক * c     = src1_তথ্য + i1 * nc;
                    ধ্রুবক দশমিক * s_row = s + i1 * ncs;
                    দশমিক         sumf  = 0.0f;
                    জন্য (int64_t i0 = 0; i0 < nc; ++i0) {
                        sumf += s_row[i0] * c[i0];
                    }
                    x[i1] = sumf;
                }
            }
        }
    }

    // Restore mask
    __asm__ অস্থির("mova.m.x %0" ::"r"(saved_mask));

    ফেরত 0;
}
