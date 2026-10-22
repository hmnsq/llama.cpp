//******************************************************************************
// L2 Norm F32 Kernel (L2 Normalization)
// y[i] = x[i] / max(sqrt(sum(x^2)), eps)
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "math_fp.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <assert.h>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <string.h>

// L2 Norm kernel parameters structure
నిర్మాణం ggml_et_l2_norm_params {
    నిర్మాణం ggml_tensor src0;  // F32 input tensor
    నిర్మాణం ggml_tensor dst;   // F32 output tensor
    దశాంశం              eps;   // Epsilon parameter for numerical stability
};

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_l2_norm_params * params, శూన్య * env) {
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
        తిరిగి -1;  // Invalid pointer
    }

    నిర్మాణం ggml_tensor * src0 = &params->src0;
    నిర్మాణం ggml_tensor * dst  = &params->dst;
    దశాంశం                eps  = params->eps;

    ఒకవేళ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి -1;  // Unsupported type combination
    }

    దశాంశం * src0_దత్తాంశం = (దశాంశం *) src0->దత్తాంశం;
    దశాంశం * dst_దత్తాంశం  = (దశాంశం *) dst->దత్తాంశం;

    ఒకవేళ (!src0_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;  // Null data pointer
    }

    ఒకవేళ (eps < 0.0f) {
        తిరిగి -1;  // Invalid epsilon
    }

    స్థిరరాశి int64_t ne0 = dst->ne[0];
    స్థిరరాశి int64_t ne1 = dst->ne[1];
    స్థిరరాశి int64_t ne2 = dst->ne[2];
    స్థిరరాశి int64_t ne3 = dst->ne[3];

    స్థిరరాశి size_t nb0 = dst->nb[0], nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    స్థిరరాశి size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];

    (శూన్య) nb0;
    (శూన్య) nb00;

    ఒకవేళ (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        తిరిగి -1;  // Shape mismatch
    }

    స్థిరరాశి int32_t మొత్తం_అడ్డువరుసలు    = (int32_t) (ne1 * ne2 * ne3);
    స్థిరరాశి పూర్ణాంకం     shire_దారాలు = SOC_MINIONS_చొప్పున_SHIRE * అంకె_HARTS_చొప్పున_MINION;

    ఒకవేళ (మొత్తం_అడ్డువరుసలు >= shire_దారాలు) {
        // Row-parallel: each thread processes whole rows
        కొరకు (int64_t i3 = 0; i3 < ne3; i3++) {
            కొరకు (int64_t i2 = 0; i2 < ne2; i2++) {
                కొరకు (int64_t i1 = thread_id; i1 < ne1; i1 += అంకె_దారాలు) {
                    స్థిరరాశి దశాంశం * src_ptr =
                        (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    దశాంశం * dst_ptr = (దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + i3 * nb3 + i2 * nb2 + i1 * nb1);

                    దశాంశం zero = 0.0f;
                    __asm__ అస్థిర("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                    కొరకు (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ అస్థిర(
                            "flw.ps f11, %[x_vec]\n"
                            "fmadd.ps f10, f11, f11, f10\n"
                            :
                            : [x_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & src_ptr[i0])
                            : "f10", "f11");
                    }

                    దశాంశం కూడిక_sq;
                    __asm__ __volatile__(
                        "fswizz.ps f1, f10, 0xB1 \n\t"
                        "fadd.ps   f2, f10, f1, rne \n\t"
                        "fswizz.ps f3, f2, 0x4E \n\t"
                        "fadd.ps   f4, f2, f3, rne \n\t"
                        "fmvz.x.ps t0, f4, 4 \n\t"
                        "fbcx.ps   f5, t0 \n\t"
                        "fadd.ps   %[vout], f4, f5, rne \n\t"
                        : [vout] "=f"(కూడిక_sq)::"t0", "f1", "f2", "f3", "f4", "f5");

                    దశాంశం l2_norm = et_powf(కూడిక_sq, 0.5f);
                    ఒకవేళ (l2_norm < eps) {
                        l2_norm = eps;
                    }
                    స్థిరరాశి దశాంశం scale = et_fdiv(1.0f, l2_norm);

                    కొరకు (int32_t i0 = 0; i0 < (int32_t) ne0; i0 += 8) {
                        __asm__ అస్థిర(
                            "flw.ps f11, %[x_vec]\n"
                            "fbc.ps f12, %[scale_ptr]\n"
                            "fmul.ps f13, f11, f12\n"
                            "fsw.ps f13, %[result]\n"
                            : [ఫలితం] "=m"(*(దశాంశం (*)[8]) & dst_ptr[i0])
                            : [x_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & src_ptr[i0]), [scale_ptr] "m"(scale)
                            : "f11", "f12", "f13");
                    }
                }
            }
        }
    } లేకపోతే {
        // Intra-row: threads within each shire cooperate via L2 SCP
        పూర్ణాంకం shire_tid       = thread_id % shire_దారాలు;
        పూర్ణాంకం దారాలు_చొప్పున_అడ్డువరుస = shire_దారాలు / మొత్తం_అడ్డువరుసలు;
        పూర్ణాంకం my_అడ్డువరుస          = shire_tid / దారాలు_చొప్పున_అడ్డువరుస;
        పూర్ణాంకం local_tid       = shire_tid % దారాలు_చొప్పున_అడ్డువరుస;
        పూర్ణాంకం సమూహం_పునాది      = my_అడ్డువరుస * దారాలు_చొప్పున_అడ్డువరుస;

        ఒకవేళ (my_అడ్డువరుస >= మొత్తం_అడ్డువరుసలు) {
            FENCE;
            et_barrier(ET_BARRIER_SHIRE);
            తిరిగి 0;
        }

        int64_t i1 = my_అడ్డువరుస % ne1;
        int64_t i2 = (my_అడ్డువరుస / ne1) % ne2;
        int64_t i3 = my_అడ్డువరుస / (ne1 * ne2);

        స్థిరరాశి దశాంశం * src_ptr = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i3 * nb03 + i2 * nb02 + i1 * nb01);
        దశాంశం *       dst_ptr = (దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + i3 * nb3 + i2 * nb2 + i1 * nb1);

        స్థిరరాశి int32_t elems_చొప్పున_cl   = 16;
        int32_t       మొత్తం_cls      = ((int32_t) ne0 + elems_చొప్పున_cl - 1) / elems_చొప్పున_cl;
        int32_t       cls_చొప్పున_thread = (మొత్తం_cls + దారాలు_చొప్పున_అడ్డువరుస - 1) / దారాలు_చొప్పున_అడ్డువరుస;
        int32_t       my_ప్రారంభం       = local_tid * cls_చొప్పున_thread * elems_చొప్పున_cl;
        int32_t       my_చివర         = my_ప్రారంభం + cls_చొప్పున_thread * elems_చొప్పున_cl;
        ఒకవేళ (my_చివర > (int32_t) ne0) {
            my_చివర = (int32_t) ne0;
        }
        ఒకవేళ (my_ప్రారంభం >= (int32_t) ne0) {
            my_ప్రారంభం = 0;
            my_చివర   = 0;
        }

        గుర్తులేని దీర్ఘ saved_ముసుగు;
        __asm__ అస్థిర("mova.x.m %0" : "=r"(saved_ముసుగు));
        __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");

        // Phase 1: partial sum of squares
        __asm__ అస్థిర("fbci.pi f10, 0" ::: "f10");
        కొరకు (int32_t i0 = my_ప్రారంభం; i0 < my_చివర; i0 += 8) {
            __asm__ అస్థిర(
                "flw.ps f11, %[x_vec]\n"
                "fmadd.ps f10, f11, f11, f10\n"
                :
                : [x_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & src_ptr[i0])
                : "f10", "f11");
        }

        దశాంశం partial_కూడిక;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(partial_కూడిక)::"t0", "f1", "f2", "f3", "f4", "f5");

        // Phase 2: L2SCP exchange
        అస్థిర దశాంశం * my_స్థానం = (అస్థిర దశాంశం *) et_shire_l2scp_local((uint64_t) shire_tid * 64);
        *my_స్థానం                 = partial_కూడిక;
        FENCE;
        evict_గమ్యం_l2((స్థిరరాశి శూన్య *) my_స్థానం, 1, 64);
        WAIT_CACHEOPS;

        et_barrier(ET_BARRIER_SHIRE);

        // Phase 3: all threads reduce + apply scale to own chunk
        పూర్ణాంకం workers = దారాలు_చొప్పున_అడ్డువరుస < మొత్తం_cls ? దారాలు_చొప్పున_అడ్డువరుస : మొత్తం_cls;

        కొరకు (పూర్ణాంకం t = 0; t < workers; t++) {
            అస్థిర దశాంశం * స్థానం = (అస్థిర దశాంశం *) et_shire_l2scp_local((uint64_t) (సమూహం_పునాది + t) * 64);
            evict_గమ్యం_l2((స్థిరరాశి శూన్య *) స్థానం, 1, 64);
        }
        WAIT_CACHEOPS;

        దశాంశం మొత్తం_కూడిక_sq = 0.0f;
        కొరకు (పూర్ణాంకం t = 0; t < workers; t++) {
            అస్థిర దశాంశం * స్థానం = (అస్థిర దశాంశం *) et_shire_l2scp_local((uint64_t) (సమూహం_పునాది + t) * 64);
            మొత్తం_కూడిక_sq += *స్థానం;
        }

        దశాంశం l2_norm = et_powf(మొత్తం_కూడిక_sq, 0.5f);
        ఒకవేళ (l2_norm < eps) {
            l2_norm = eps;
        }
        స్థిరరాశి దశాంశం scale = et_fdiv(1.0f, l2_norm);

        ఒకవేళ (my_ప్రారంభం < my_చివర) {
            uint32_t scale_బిట్లు;
            __asm__ అస్థిర("fmv.x.s %0, %1" : "=r"(scale_బిట్లు) : "f"(scale));
            __asm__ అస్థిర("fbcx.ps f13, %[sb]\n" : : [sb] "r"(scale_బిట్లు) : "f13");

            కొరకు (int32_t i0 = my_ప్రారంభం; i0 < my_చివర; i0 += 8) {
                __asm__ అస్థిర(
                    "flw.ps f12, %[x_vec]\n"
                    "fmul.ps f14, f12, f13\n"
                    "fsw.ps f14, %[result]\n"
                    : [ఫలితం] "=m"(*(దశాంశం (*)[8]) & dst_ptr[i0])
                    : [x_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & src_ptr[i0])
                    : "f12", "f14");
            }
        }

        __asm__ అస్థిర("mova.m.x %0" ::"r"(saved_ముసుగు));
    }

    తిరిగి 0;
}
