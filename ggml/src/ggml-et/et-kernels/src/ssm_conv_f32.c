#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>

నిర్మాణం ggml_et_ssm_conv_params {
    నిర్మాణం ggml_tensor src0;  // conv_x: [d_conv - 1 + n_t, d_inner, n_seqs]
    నిర్మాణం ggml_tensor src1;  // conv1d.weight: [d_conv, d_inner]
    నిర్మాణం ggml_tensor dst;   // output: [d_inner, n_t, n_seqs]
};

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_ssm_conv_params * params, శూన్య * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ఒకవేళ (!kernel_env) {
        తిరిగి -1;
    }

    పూర్ణాంకం thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    పూర్ణాంకం అంకె_దారాలు = get_అంకె_దారాలు(kernel_env->shire_ముసుగు);

    ఒకవేళ (thread_id < 0) {
        తిరిగి 0;
    }

    ఒకవేళ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        తిరిగి -1;
    }

    నిర్మాణం ggml_tensor * src0 = &params->src0;
    నిర్మాణం ggml_tensor * src1 = &params->src1;
    నిర్మాణం ggml_tensor * dst  = &params->dst;

    ఒకవేళ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి -1;
    }

    స్థిరరాశి దశాంశం * src0_దత్తాంశం = (స్థిరరాశి దశాంశం *) src0->దత్తాంశం;
    స్థిరరాశి దశాంశం * src1_దత్తాంశం = (స్థిరరాశి దశాంశం *) src1->దత్తాంశం;
    దశాంశం *       dst_దత్తాంశం  = (దశాంశం *) dst->దత్తాంశం;

    ఒకవేళ (!src0_దత్తాంశం || !src1_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t nc  = src1->ne[0];
    స్థిరరాశి int64_t ncs = src0->ne[0];
    స్థిరరాశి int64_t nr  = src0->ne[1];
    స్థిరరాశి int64_t n_t = dst->ne[1];
    స్థిరరాశి int64_t n_s = dst->ne[2];

    ఒకవేళ (dst->ne[0] != nr || src1->ne[1] != nr || ncs != nc - 1 + n_t || src0->nb[0] != పరిమాణం(దశాంశం) ||
        src1->nb[0] != పరిమాణం(దశాంశం) || dst->nb[0] != పరిమాణం(దశాంశం) || src0->nb[1] != (size_t) ncs * పరిమాణం(దశాంశం) ||
        src1->nb[1] != (size_t) nc * పరిమాణం(దశాంశం)) {
        తిరిగి -1;
    }

    // Parallelize over d_inner in cache-line-aligned chunks (16 floats = 64B)
    స్థిరరాశి int64_t chunk    = 16;
    స్థిరరాశి int64_t n_chunks = (nr + chunk - 1) / chunk;

    // Save and set vector mask to all 8 lanes
    గుర్తులేని దీర్ఘ saved_ముసుగు;
    __asm__ అస్థిర("mova.x.m %0" : "=r"(saved_ముసుగు));
    __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");

    కొరకు (int64_t i3 = 0; i3 < n_s; ++i3) {
        కొరకు (int64_t i2 = 0; i2 < n_t; ++i2) {
            స్థిరరాశి దశాంశం * s = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i2 * src0->nb[0] + i3 * src0->nb[2]);
            దశాంశం *       x = (దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + i2 * dst->nb[1] + i3 * dst->nb[2]);

            కొరకు (int64_t ci = thread_id; ci < n_chunks; ci += అంకె_దారాలు) {
                స్థిరరాశి int64_t i1_ప్రారంభం = ci * chunk;
                స్థిరరాశి int64_t i1_చివర   = i1_ప్రారంభం + chunk < nr ? i1_ప్రారంభం + chunk : nr;

                // Process 8 channels at a time with SIMD
                int64_t i1 = i1_ప్రారంభం;
                కొరకు (; i1 + 8 <= i1_చివర; i1 += 8) {
                    // Gather 8 channels' data into contiguous buffers for each tap
                    దశాంశం tmp_s[8], tmp_c[8];
                    దశాంశం acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

                    కొరకు (int64_t i0 = 0; i0 < nc; ++i0) {
                        // TODO: Some way to get rid of this gather
                        కొరకు (పూర్ణాంకం j = 0; j < 8; ++j) {
                            tmp_s[j] = s[(i1 + j) * ncs + i0];
                            tmp_c[j] = src1_దత్తాంశం[(i1 + j) * nc + i0];
                        }

                        __asm__ అస్థిర(
                            "flw.ps f10, %[acc]\n"
                            "flw.ps f11, %[sv]\n"
                            "flw.ps f12, %[cv]\n"
                            "fmadd.ps f10, f11, f12, f10\n"
                            "fsw.ps f10, %[out]\n"
                            : [బహిర్గతం] "=m"(*(దశాంశం (*)[8]) acc)
                            : [acc] "m"(*(స్థిరరాశి దశాంశం (*)[8]) acc), [sv] "m"(*(స్థిరరాశి దశాంశం (*)[8]) tmp_s),
                              [cv] "m"(*(స్థిరరాశి దశాంశం (*)[8]) tmp_c)
                            : "f10", "f11", "f12");
                    }

                    // Store 8 results — dst is contiguous along d_inner
                    __asm__ అస్థిర(
                        "flw.ps f10, %[acc]\n"
                        "fsw.ps f10, %[dst]\n"
                        : [dst] "=m"(*(దశాంశం (*)[8])(x + i1))
                        : [acc] "m"(*(స్థిరరాశి దశాంశం (*)[8]) acc)
                        : "f10");
                }

                // Scalar tail for remaining channels
                కొరకు (; i1 < i1_చివర; ++i1) {
                    స్థిరరాశి దశాంశం * c     = src1_దత్తాంశం + i1 * nc;
                    స్థిరరాశి దశాంశం * s_అడ్డువరుస = s + i1 * ncs;
                    దశాంశం         sumf  = 0.0f;
                    కొరకు (int64_t i0 = 0; i0 < nc; ++i0) {
                        sumf += s_అడ్డువరుస[i0] * c[i0];
                    }
                    x[i1] = sumf;
                }
            }
        }
    }

    // Restore mask
    __asm__ అస్థిర("mova.m.x %0" ::"r"(saved_ముసుగు));

    తిరిగి 0;
}
