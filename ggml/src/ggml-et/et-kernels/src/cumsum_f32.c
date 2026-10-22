//******************************************************************************
// CUMSUM F32 Kernel
// Computes an inclusive prefix sum along dim 0 for each row in higher dims.
// First-pass implementation: scalar and row-contiguous input/output only.
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>

నిర్మాణం ggml_et_cumsum_params {
    నిర్మాణం ggml_tensor src0;
    నిర్మాణం ggml_tensor dst;
};

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_cumsum_params * params, శూన్య * env) {
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

    దశాంశం * src0_దత్తాంశం = (దశాంశం *) src0->దత్తాంశం;
    దశాంశం * dst_దత్తాంశం  = (దశాంశం *) dst->దత్తాంశం;

    ఒకవేళ (!src0_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t ne0 = src0->ne[0];
    స్థిరరాశి int64_t ne1 = src0->ne[1];
    స్థిరరాశి int64_t ne2 = src0->ne[2];
    స్థిరరాశి int64_t ne3 = src0->ne[3];

    స్థిరరాశి size_t snb0 = src0->nb[0];
    స్థిరరాశి size_t snb1 = src0->nb[1];
    స్థిరరాశి size_t snb2 = src0->nb[2];
    స్థిరరాశి size_t snb3 = src0->nb[3];

    స్థిరరాశి size_t dnb0 = dst->nb[0];
    స్థిరరాశి size_t dnb1 = dst->nb[1];
    స్థిరరాశి size_t dnb2 = dst->nb[2];
    స్థిరరాశి size_t dnb3 = dst->nb[3];

    ఒకవేళ (snb0 != పరిమాణం(దశాంశం) || dnb0 != పరిమాణం(దశాంశం)) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t మొత్తం_అడ్డువరుసలు     = ne1 * ne2 * ne3;
    స్థిరరాశి int64_t అడ్డువరుసలు_చొప్పున_సమూహం = et_అడ్డువరుసలు_చొప్పున_cacheline_సమూహం(ne0, పరిమాణం(దశాంశం));
    స్థిరరాశి int64_t మొత్తం_groups   = (మొత్తం_అడ్డువరుసలు + అడ్డువరుసలు_చొప్పున_సమూహం - 1) / అడ్డువరుసలు_చొప్పున_సమూహం;

    కొరకు (int64_t grp = thread_id; grp < మొత్తం_groups; grp += అంకె_దారాలు) {
        స్థిరరాశి int64_t అడ్డువరుస_ప్రారంభం = grp * అడ్డువరుసలు_చొప్పున_సమూహం;
        int64_t       అడ్డువరుస_చివర   = అడ్డువరుస_ప్రారంభం + అడ్డువరుసలు_చొప్పున_సమూహం;
        ఒకవేళ (అడ్డువరుస_చివర > మొత్తం_అడ్డువరుసలు) {
            అడ్డువరుస_చివర = మొత్తం_అడ్డువరుసలు;
        }

        కొరకు (int64_t అడ్డువరుస = అడ్డువరుస_ప్రారంభం; అడ్డువరుస < అడ్డువరుస_చివర; ++అడ్డువరుస) {
            int64_t i1 = అడ్డువరుస % ne1;
            int64_t i2 = (అడ్డువరుస / ne1) % ne2;
            int64_t i3 = అడ్డువరుస / (ne1 * ne2);

            స్థిరరాశి దశాంశం * src_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i1 * snb1 + i2 * snb2 + i3 * snb3);
            దశాంశం *       dst_అడ్డువరుస = (దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

            దశాంశం acc = 0.0f;
            కొరకు (int64_t i0 = 0; i0 < ne0; ++i0) {
                acc += src_అడ్డువరుస[i0];
                dst_అడ్డువరుస[i0] = acc;
            }
        }
    }

    తిరిగి 0;
}
