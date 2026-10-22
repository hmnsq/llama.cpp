//******************************************************************************
// SUM_ROWS F32 Kernel
// Row-wise sum reduction: dst[0, i1, i2, i3] = sum(src0[0..ne00-1, i1, i2, i3])
// Vectorized 8-wide accumulation with horizontal reduction.
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>

నిర్మాణం ggml_et_కూడిక_అడ్డువరుసలు_params {
    నిర్మాణం ggml_tensor src0;  // F32 input tensor [ne00, ne01, ne02, ne03]
    నిర్మాణం ggml_tensor dst;   // F32 output tensor [1, ne01, ne02, ne03]
};

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_కూడిక_అడ్డువరుసలు_params * params, శూన్య * env) {
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


    స్థిరరాశి int64_t ne00 = src0->ne[0];  // Row length (to be summed)
    స్థిరరాశి int64_t ne01 = src0->ne[1];
    స్థిరరాశి int64_t ne02 = src0->ne[2];
    స్థిరరాశి int64_t ne03 = src0->ne[3];

    స్థిరరాశి size_t nb01 = src0->nb[1];
    స్థిరరాశి size_t nb02 = src0->nb[2];
    స్థిరరాశి size_t nb03 = src0->nb[3];

    స్థిరరాశి size_t nb1 = dst->nb[1];
    స్థిరరాశి size_t nb2 = dst->nb[2];
    స్థిరరాశి size_t nb3 = dst->nb[3];

    // Flatten rows across dimensions 1,2,3 and distribute across threads
    స్థిరరాశి int64_t మొత్తం_అడ్డువరుసలు = ne01 * ne02 * ne03;

    కొరకు (int64_t ir = thread_id; ir < మొత్తం_అడ్డువరుసలు; ir += అంకె_దారాలు) {
        స్థిరరాశి int64_t i03 = ir / (ne02 * ne01);
        స్థిరరాశి int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
        స్థిరరాశి int64_t i01 = ir - i03 * ne02 * ne01 - i02 * ne01;

        స్థిరరాశి దశాంశం * src_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i01 * nb01 + i02 * nb02 + i03 * nb03);
        దశాంశం *       dst_ptr = (దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + i01 * nb1 + i02 * nb2 + i03 * nb3);

        // Vectorized 8-wide sum accumulation
        దశాంశం zero = 0.0f;
        __asm__ అస్థిర("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        కొరకు (int32_t i0 = 0; i0 < (int32_t) ne00; i0 += 8) {
            __asm__ అస్థిర(
                "flw.ps f11, %[x_vec]\n"
                "fadd.ps f10, f10, f11\n"
                :
                : [x_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & src_అడ్డువరుస[i0])
                : "f10", "f11");
        }

        // Horizontal sum of 8 accumulated values in f10
        దశాంశం అడ్డువరుస_కూడిక;
        __asm__ __volatile__(
            "fswizz.ps f1, f10, 0xB1 \n\t"
            "fadd.ps   f2, f10, f1, rne \n\t"
            "fswizz.ps f3, f2, 0x4E \n\t"
            "fadd.ps   f4, f2, f3, rne \n\t"
            "fmvz.x.ps t0, f4, 4 \n\t"
            "fbcx.ps   f5, t0 \n\t"
            "fadd.ps   %[vout], f4, f5, rne \n\t"
            : [vout] "=f"(అడ్డువరుస_కూడిక)::"t0", "f1", "f2", "f3", "f4", "f5");

        atomic_store_f32(dst_ptr, అడ్డువరుస_కూడిక);
    }

    తిరిగి 0;
}
