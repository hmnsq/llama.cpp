//******************************************************************************
// Bare Metal PAD F32 Kernel
// Zero-pads an F32 tensor along dimensions 1-3.
//
// Constraints:
//   - No dim0 padding (lp[0]==0, rp[0]==0)
//   - dst contiguous
//   - src nb[0] == 4 (dim0 contiguous for vectorized reads)
//   - Zero-pad only (no circular mode)
//
// Two paths:
//   Aligned (ne0 % 16 == 0): rows distributed across harts, vectorized.
//   Small   (16 % ne0 == 0): cache-line distributed, scalar per-element.
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>

నిర్మాణం ggml_et_pad_params {
    నిర్మాణం ggml_tensor src0;
    నిర్మాణం ggml_tensor dst;
    int32_t            lp[4];
    int32_t            rp[4];
};

// Vectorized copy with scalar tail
స్థిర అంతఃవరుస శూన్య vec_copy_f32(దశాంశం * dst, స్థిరరాశి దశాంశం * src, int32_t n) {
    int32_t       i       = 0;
    స్థిరరాశి int32_t vec_చివర = (n / 8) * 8;
    కొరకు (; i < vec_చివర; i += 8) {
        __asm__ అస్థిర(
            "flw.ps f10, %[s]\n"
            "fsw.ps f10, %[d]\n"
            : [d] "=m"(*(దశాంశం (*)[8]) & dst[i])
            : [s] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & src[i])
            : "f10");
    }
    కొరకు (; i < n; i++) {
        dst[i] = src[i];
    }
}

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_pad_params * params, శూన్య * env) {
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
    నిర్మాణం ggml_tensor * dst  = &params->dst;

    ఒకవేళ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి -1;
    }

    స్థిరరాశి దశాంశం * src0_దత్తాంశం = (స్థిరరాశి దశాంశం *) src0->దత్తాంశం;
    దశాంశం *       dst_దత్తాంశం  = (దశాంశం *) dst->దత్తాంశం;

    ఒకవేళ (!src0_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;
    }

    // Dst dimensions
    స్థిరరాశి int64_t ne0 = dst->ne[0];
    స్థిరరాశి int64_t ne1 = dst->ne[1];
    స్థిరరాశి int64_t ne2 = dst->ne[2];
    స్థిరరాశి int64_t ne3 = dst->ne[3];

    // Src strides (byte offsets)
    స్థిరరాశి int64_t nb1_src = src0->nb[1];
    స్థిరరాశి int64_t nb2_src = src0->nb[2];
    స్థిరరాశి int64_t nb3_src = src0->nb[3];

    // Padding values
    స్థిరరాశి int32_t lp1 = params->lp[1];
    స్థిరరాశి int32_t rp1 = params->rp[1];
    స్థిరరాశి int32_t lp2 = params->lp[2];
    స్థిరరాశి int32_t rp2 = params->rp[2];
    స్థిరరాశి int32_t lp3 = params->lp[3];
    స్థిరరాశి int32_t rp3 = params->rp[3];

    స్థిరరాశి int64_t మొత్తం_అడ్డువరుసలు     = ne1 * ne2 * ne3;
    స్థిరరాశి int64_t మొత్తం_elements = ne0 * మొత్తం_అడ్డువరుసలు;

    ఒకవేళ (మొత్తం_elements == 0) {
        తిరిగి 0;
    }

    // Broadcast 0.0f to SIMD register for vectorized zero-fill
    దశాంశం zero = 0.0f;
    __asm__ అస్థిర("fbc.ps f12, %[v]\n" : : [v] "m"(zero) : "f12");

    // Aligned: ne0 % 16 == 0 -> row-based distribution, vectorized
    ఒకవేళ (ne0 % 16 == 0) {
        కొరకు (int64_t అడ్డువరుస = thread_id; అడ్డువరుస < మొత్తం_అడ్డువరుసలు; అడ్డువరుస += అంకె_దారాలు) {
            స్థిరరాశి int64_t i3 = అడ్డువరుస / (ne1 * ne2);
            స్థిరరాశి int64_t i2 = (అడ్డువరుస / ne1) % ne2;
            స్థిరరాశి int64_t i1 = అడ్డువరుస % ne1;

            దశాంశం * dst_అడ్డువరుస = dst_దత్తాంశం + అడ్డువరుస * ne0;

            ఒకవేళ (i1 >= lp1 && i1 < ne1 - rp1 && i2 >= lp2 && i2 < ne2 - rp2 && i3 >= lp3 && i3 < ne3 - rp3) {
                స్థిరరాశి దశాంశం * src_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + (i1 - lp1) * nb1_src +
                                                         (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                vec_copy_f32(dst_అడ్డువరుస, src_అడ్డువరుస, (int32_t) ne0);
            } లేకపోతే {
                int64_t       i       = 0;
                స్థిరరాశి int64_t vec_చివర = (ne0 / 8) * 8;
                కొరకు (; i < vec_చివర; i += 8) {
                    __asm__ అస్థిర("fsw.ps f12, %[d]\n" : [d] "=m"(*(దశాంశం (*)[8]) & dst_అడ్డువరుస[i])::"f12");
                }
            }
        }
        తిరిగి 0;
    }

    // Small-ne0 path: 16 % ne0 == 0 -> cache-line distributed, scalar
    స్థిరరాశి int64_t elems_చొప్పున_cl = 16;
    స్థిరరాశి int64_t మొత్తం_cl     = (మొత్తం_elements + elems_చొప్పున_cl - 1) / elems_చొప్పున_cl;

    స్థిరరాశి int64_t ne1_దత్తాంశం_చివర = ne1 - rp1;
    స్థిరరాశి int64_t ne2_దత్తాంశం_చివర = ne2 - rp2;
    స్థిరరాశి int64_t ne3_దత్తాంశం_చివర = ne3 - rp3;

    కొరకు (int64_t cl = thread_id; cl < మొత్తం_cl; cl += అంకె_దారాలు) {
        స్థిరరాశి int64_t elem_ప్రారంభం = cl * elems_చొప్పున_cl;
        int64_t       elem_చివర   = elem_ప్రారంభం + elems_చొప్పున_cl;
        ఒకవేళ (elem_చివర > మొత్తం_elements) {
            elem_చివర = మొత్తం_elements;
        }

        కొరకు (int64_t idx = elem_ప్రారంభం; idx < elem_చివర; idx++) {
            స్థిరరాశి int64_t i0   = idx % ne0;
            స్థిరరాశి int64_t rem  = idx / ne0;
            స్థిరరాశి int64_t i1   = rem % ne1;
            స్థిరరాశి int64_t rem2 = rem / ne1;
            స్థిరరాశి int64_t i2   = rem2 % ne2;
            స్థిరరాశి int64_t i3   = rem2 / ne2;

            ఒకవేళ (i1 >= lp1 && i1 < ne1_దత్తాంశం_చివర && i2 >= lp2 && i2 < ne2_దత్తాంశం_చివర && i3 >= lp3 && i3 < ne3_దత్తాంశం_చివర) {
                స్థిరరాశి దశాంశం * sp = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i0 * 4 + (i1 - lp1) * nb1_src +
                                                    (i2 - lp2) * nb2_src + (i3 - lp3) * nb3_src);
                dst_దత్తాంశం[idx]    = *sp;
            } లేకపోతే {
                dst_దత్తాంశం[idx] = 0.0f;
            }
        }
    }

    తిరిగి 0;
}
