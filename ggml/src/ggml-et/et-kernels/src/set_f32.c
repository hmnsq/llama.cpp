//******************************************************************************
// SET F32 Kernel
// Minimal ET implementation for inplace F32 SET into a contiguous destination
// using a contiguous F32 source view and explicit destination view strides.
//
// Supported shape family:
// - dst/base is contiguous F32
// - src1 is contiguous F32
// - src1.ne[0] is cacheline-aligned (multiple of 16 floats)
// - destination view strides/offset are cacheline-aligned
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>

నిర్మాణం ggml_et_సమితి_params {
    నిర్మాణం ggml_tensor src1;
    నిర్మాణం ggml_tensor dst;
    int32_t            nb1;
    int32_t            nb2;
    int32_t            nb3;
    int32_t            స్థానభ్రంశం;
};

స్థిర అంతఃవరుస శూన్య copy_అడ్డువరుస_aligned(దశాంశం * dst, స్థిరరాశి దశాంశం * src, int32_t n) {
    కొరకు (int32_t i = 0; i < n; i += 8) {
        __asm__ అస్థిర(
            "flw.ps f11, %[src_vec]\n"
            "fsw.ps f11, %[dst_vec]\n"
            : [dst_vec] "=m"(*(దశాంశం (*)[8]) & dst[i])
            : [src_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & src[i])
            : "f11");
    }
}

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_సమితి_params * params, శూన్య * env) {
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

    నిర్మాణం ggml_tensor * src1 = &params->src1;
    నిర్మాణం ggml_tensor * dst  = &params->dst;

    ఒకవేళ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి -1;
    }

    స్థిరరాశి దశాంశం * src1_దత్తాంశం = (స్థిరరాశి దశాంశం *) src1->దత్తాంశం;
    దశాంశం *       dst_దత్తాంశం  = (దశాంశం *) dst->దత్తాంశం;
    ఒకవేళ (!src1_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t ne10 = src1->ne[0];
    స్థిరరాశి int64_t ne11 = src1->ne[1];
    స్థిరరాశి int64_t ne12 = src1->ne[2];
    స్థిరరాశి int64_t ne13 = src1->ne[3];

    ఒకవేళ (src1->nb[0] != పరిమాణం(దశాంశం) || dst->nb[0] != పరిమాణం(దశాంశం) || ne10 % 16 != 0) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t nb11 = src1->nb[1];
    స్థిరరాశి int64_t nb12 = src1->nb[2];
    స్థిరరాశి int64_t nb13 = src1->nb[3];

    స్థిరరాశి int64_t dnb1   = params->nb1;
    స్థిరరాశి int64_t dnb2   = params->nb2;
    స్థిరరాశి int64_t dnb3   = params->nb3;
    స్థిరరాశి int64_t స్థానభ్రంశం = params->స్థానభ్రంశం;

    స్థిరరాశి int64_t మొత్తం_అడ్డువరుసలు = ne11 * ne12 * ne13;

    కొరకు (int64_t అడ్డువరుస = thread_id; అడ్డువరుస < మొత్తం_అడ్డువరుసలు; అడ్డువరుస += అంకె_దారాలు) {
        స్థిరరాశి int64_t i1 = అడ్డువరుస % ne11;
        స్థిరరాశి int64_t i2 = (అడ్డువరుస / ne11) % ne12;
        స్థిరరాశి int64_t i3 = అడ్డువరుస / (ne11 * ne12);

        స్థిరరాశి దశాంశం * src_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1_దత్తాంశం + i1 * nb11 + i2 * nb12 + i3 * nb13);
        దశాంశం *       dst_అడ్డువరుస = (దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + స్థానభ్రంశం + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        copy_అడ్డువరుస_aligned(dst_అడ్డువరుస, src_అడ్డువరుస, (int32_t) ne10);
    }

    తిరిగి 0;
}
