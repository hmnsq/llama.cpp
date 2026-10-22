//******************************************************************************
// Diag F32 Kernel
// Creates a diagonal matrix from a 1D vector.
// dst[i][j] = (i == j) ? src0[i] : 0.0f
//
// src0: [N, 1, ne2, ne3]   (1D vector per batch)
// dst:  [N, N, ne2, ne3]   (diagonal matrix per batch)
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>

నిర్మాణం ggml_et_diag_params {
    నిర్మాణం ggml_tensor src0;  // F32 input vector
    నిర్మాణం ggml_tensor dst;   // F32 output diagonal matrix
};

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_diag_params * params, శూన్య * env) {
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

    స్థిరరాశి int64_t ne0 = dst->ne[0];  // N (row width = column count)
    స్థిరరాశి int64_t ne1 = dst->ne[1];  // N (number of rows)
    స్థిరరాశి int64_t ne2 = dst->ne[2];
    స్థిరరాశి int64_t ne3 = dst->ne[3];

    స్థిరరాశి size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];
    స్థిరరాశి size_t nb02 = src0->nb[2], nb03 = src0->nb[3];

    // Total rows across all batches — parallelize over these
    స్థిరరాశి int64_t మొత్తం_అడ్డువరుసలు = ne1 * ne2 * ne3;

    // Prepare zero vector for SIMD zeroing
    దశాంశం zero = 0.0f;
    __asm__ అస్థిర("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

    కొరకు (int64_t అడ్డువరుస = thread_id; అడ్డువరుస < మొత్తం_అడ్డువరుసలు; అడ్డువరుస += అంకె_దారాలు) {
        int64_t i1 = అడ్డువరుస % ne1;
        int64_t i2 = (అడ్డువరుస / ne1) % ne2;
        int64_t i3 = అడ్డువరుస / (ne1 * ne2);

        దశాంశం * dst_అడ్డువరుస = (దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + i1 * nb1 + i2 * nb2 + i3 * nb3);

        // Zero the entire row with SIMD
        int64_t       i0      = 0;
        స్థిరరాశి int64_t vec_చివర = (ne0 / 8) * 8;
        కొరకు (; i0 < vec_చివర; i0 += 8) {
            __asm__ అస్థిర("fsw.ps f10, %[d]\n" : [d] "=m"(*(దశాంశం (*)[8]) & dst_అడ్డువరుస[i0])::"f10");
        }
        కొరకు (; i0 < ne0; i0++) {
            dst_అడ్డువరుస[i0] = 0.0f;
        }

        // Place the diagonal element: dst[i1][i1] = src0[i1]
        స్థిరరాశి దశాంశం * src_ptr = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i2 * nb02 + i3 * nb03);
        dst_అడ్డువరుస[i1]           = src_ptr[i1];
    }

    తిరిగి 0;
}
