//******************************************************************************
// Repeat F32 Kernel
// Tiles src0 into dst: dst.ne[i] = src0.ne[i] * nr[i] for each dimension.
// All copies are cacheline-aligned (ne00 % 16 == 0).
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>
#అన్తర్భావ <string.h>

నిర్మాణం ggml_et_repeat_params {
    నిర్మాణం ggml_tensor src0;  // F32 input tensor (tile)
    నిర్మాణం ggml_tensor dst;   // F32 output tensor (tiled result)
};

// Copy n floats from src to dst using 8-wide vector loads/stores.
// n must be a multiple of 16 (cacheline-aligned).
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

// Broadcast a single scalar to n floats using fbc.ps (broadcast to all lanes).
// n must be a multiple of 16 (cacheline-aligned).
స్థిర అంతఃవరుస శూన్య ప్రసారం_scalar_aligned(దశాంశం * dst, దశాంశం val, int32_t n) {
    __asm__ అస్థిర("fbc.ps f11, %[v]\n" : : [v] "m"(val) : "f11");
    కొరకు (int32_t i = 0; i < n; i += 8) {
        __asm__ అస్థిర("fsw.ps f11, %[dst_vec]\n" : [dst_vec] "=m"(*(దశాంశం (*)[8]) & dst[i])::"f11");
    }
}

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_repeat_params * params, శూన్య * env) {
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

    స్థిరరాశి int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    స్థిరరాశి int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];

    // src0 strides in bytes
    స్థిరరాశి size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    // dst strides in bytes
    స్థిరరాశి size_t dnb0 = dst->nb[0], dnb1 = dst->nb[1], dnb2 = dst->nb[2], dnb3 = dst->nb[3];

    // Repeat counts per dimension
    స్థిరరాశి int32_t nr0 = (int32_t) (ne0 / ne00);
    స్థిరరాశి int32_t nr1 = (int32_t) (ne1 / ne01);
    స్థిరరాశి int32_t nr2 = (int32_t) (ne2 / ne02);
    స్థిరరాశి int32_t nr3 = (int32_t) (ne3 / ne03);

    // Total output rows across all dimensions (excluding dim 0 tiling)
    స్థిరరాశి int64_t మొత్తం_అడ్డువరుసలు = ne1 * ne2 * ne3;

    కొరకు (int64_t అడ్డువరుస = thread_id; అడ్డువరుస < మొత్తం_అడ్డువరుసలు; అడ్డువరుస += అంకె_దారాలు) {
        // Decompose linear row index into dst (i1, i2, i3)
        int64_t i1 = అడ్డువరుస % ne1;
        int64_t i2 = (అడ్డువరుస / ne1) % ne2;
        int64_t i3 = అడ్డువరుస / (ne1 * ne2);

        // Map dst indices back to src0 indices (modular wrap)
        int64_t k1 = i1 % ne01;
        int64_t k2 = i2 % ne02;
        int64_t k3 = i3 % ne03;

        స్థిరరాశి దశాంశం * src_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + k1 * nb01 + k2 * nb02 + k3 * nb03);
        దశాంశం *       dst_అడ్డువరుస = (దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        ఒకవేళ (ne00 == 1) {
            // Scalar broadcast: splat single value across entire dst row
            ప్రసారం_scalar_aligned(dst_అడ్డువరుస, *src_అడ్డువరుస, (int32_t) ne0);
        } లేకపోతే ఒకవేళ (nr0 == 1) {
            // No tiling along dim 0 - single cacheline-aligned row copy
            copy_అడ్డువరుస_aligned(dst_అడ్డువరుస, src_అడ్డువరుస, (int32_t) ne00);
        } లేకపోతే {
            // Tile ne00-sized chunks across dim 0
            కొరకు (int32_t i0 = 0; i0 < nr0; i0++) {
                copy_అడ్డువరుస_aligned(dst_అడ్డువరుస + i0 * ne00, src_అడ్డువరుస, (int32_t) ne00);
            }
        }
    }

    తిరిగి 0;
}
