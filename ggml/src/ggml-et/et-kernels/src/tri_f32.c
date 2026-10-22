//******************************************************************************
// Tri F32 Kernel
// Triangular masking: zero out elements outside the triangular region.
//
// tri_type (matches ggml_tri_type enum):
//   0 = UPPER_DIAG:  keep where i0 >= i1
//   1 = UPPER:       keep where i0 >  i1
//   2 = LOWER_DIAG:  keep where i0 <= i1
//   3 = LOWER:       keep where i0 <  i1
//
// Distribution: cache-line aligned chunks of the flat contiguous dst.
// Each element is individually classified as keep or zero based on its
// (i0, i1) coordinates. This avoids cache-line sharing between threads
// when ne0 is not a multiple of 16.
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>

#నిర్వచించు TRI_TYPE_పై_DIAG 0
#నిర్వచించు TRI_TYPE_పై      1
#నిర్వచించు TRI_TYPE_క్రింది_DIAG 2
#నిర్వచించు TRI_TYPE_క్రింది      3

నిర్మాణం ggml_et_tri_params {
    నిర్మాణం ggml_tensor src0;
    నిర్మాణం ggml_tensor dst;
    int32_t            tri_type;
};

స్థిర అంతఃవరుస పూర్ణాంకం keep_element(int32_t tri_type, int64_t i0, int64_t i1) {
    ఎంపిక (tri_type) {
        స్థితి TRI_TYPE_క్రింది:
            తిరిగి i0 < i1;
        స్థితి TRI_TYPE_క్రింది_DIAG:
            తిరిగి i0 <= i1;
        స్థితి TRI_TYPE_పై:
            తిరిగి i0 > i1;
        స్థితి TRI_TYPE_పై_DIAG:
            తిరిగి i0 >= i1;
        అప్రమేయ:
            తిరిగి 0;
    }
}

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_tri_params * params, శూన్య * env) {
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

    నిర్మాణం ggml_tensor * src0     = &params->src0;
    నిర్మాణం ggml_tensor * dst      = &params->dst;
    int32_t              tri_type = params->tri_type;

    ఒకవేళ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి -1;
    }

    దశాంశం * src0_దత్తాంశం = (దశాంశం *) src0->దత్తాంశం;
    దశాంశం * dst_దత్తాంశం  = (దశాంశం *) dst->దత్తాంశం;

    ఒకవేళ (!src0_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t ne0 = dst->ne[0];
    స్థిరరాశి int64_t ne1 = dst->ne[1];
    స్థిరరాశి int64_t ne2 = dst->ne[2];
    స్థిరరాశి int64_t ne3 = dst->ne[3];

    స్థిరరాశి size_t nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    స్థిరరాశి size_t nb1 = dst->nb[1], nb2 = dst->nb[2], nb3 = dst->nb[3];

    స్థిరరాశి int64_t మొత్తం_అడ్డువరుసలు = ne1 * ne2 * ne3;

    //==========================================================================
    // Fast path: ne0 % 16 == 0 — rows are cache-line aligned, distribute rows
    //==========================================================================
    ఒకవేళ (ne0 % 16 == 0) {
        దశాంశం zero = 0.0f;
        __asm__ అస్థిర("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

        కొరకు (int64_t అడ్డువరుస = thread_id; అడ్డువరుస < మొత్తం_అడ్డువరుసలు; అడ్డువరుస += అంకె_దారాలు) {
            స్థిరరాశి int64_t i1 = అడ్డువరుస % ne1;
            స్థిరరాశి int64_t i2 = (అడ్డువరుస / ne1) % ne2;
            స్థిరరాశి int64_t i3 = అడ్డువరుస / (ne1 * ne2);

            స్థిరరాశి దశాంశం * src_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i1 * nb01 + i2 * nb02 + i3 * nb03);
            దశాంశం *       dst_అడ్డువరుస = (దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + i1 * nb1 + i2 * nb2 + i3 * nb3);

            int64_t keep_ప్రారంభం, keep_చివర;
            ఎంపిక (tri_type) {
                స్థితి TRI_TYPE_క్రింది:
                    keep_ప్రారంభం = 0;
                    keep_చివర   = i1;
                    ఆపు;
                స్థితి TRI_TYPE_క్రింది_DIAG:
                    keep_ప్రారంభం = 0;
                    keep_చివర   = i1 + 1;
                    ఆపు;
                స్థితి TRI_TYPE_పై:
                    keep_ప్రారంభం = i1 + 1;
                    keep_చివర   = ne0;
                    ఆపు;
                స్థితి TRI_TYPE_పై_DIAG:
                    keep_ప్రారంభం = i1;
                    keep_చివర   = ne0;
                    ఆపు;
                అప్రమేయ:
                    తిరిగి -1;
            }
            ఒకవేళ (keep_చివర > ne0) {
                keep_చివర = ne0;
            }

            // Zero prefix [0, keep_start) — SIMD for aligned blocks, scalar tail
            int64_t i0 = 0;
            కొరకు (; i0 + 8 <= keep_ప్రారంభం; i0 += 8) {
                __asm__ అస్థిర("fsw.ps f10, %[d]\n" : [d] "=m"(*(దశాంశం (*)[8]) & dst_అడ్డువరుస[i0])::"f10");
            }
            కొరకు (; i0 < keep_ప్రారంభం; i0++) {
                dst_అడ్డువరుస[i0] = 0.0f;
            }

            // Copy kept region [keep_start, keep_end) — SIMD + scalar tail
            కొరకు (; i0 + 8 <= keep_చివర; i0 += 8) {
                __asm__ అస్థిర(
                    "flw.ps f11, %[s]\n"
                    "fsw.ps f11, %[d]\n"
                    : [d] "=m"(*(దశాంశం (*)[8]) & dst_అడ్డువరుస[i0])
                    : [s] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & src_అడ్డువరుస[i0])
                    : "f11");
            }
            కొరకు (; i0 < keep_చివర; i0++) {
                dst_అడ్డువరుస[i0] = src_అడ్డువరుస[i0];
            }

            // Zero suffix [keep_end, ne0) — SIMD + scalar tail
            కొరకు (; i0 + 8 <= ne0; i0 += 8) {
                __asm__ అస్థిర("fsw.ps f10, %[d]\n" : [d] "=m"(*(దశాంశం (*)[8]) & dst_అడ్డువరుస[i0])::"f10");
            }
            కొరకు (; i0 < ne0; i0++) {
                dst_అడ్డువరుస[i0] = 0.0f;
            }
        }
        తిరిగి 0;
    }

    //==========================================================================
    // Unaligned fallback: distribute by cache lines, scalar per element
    //==========================================================================
    {
        స్థిరరాశి int64_t మొత్తం_elements = ne0 * ne1 * ne2 * ne3;
        స్థిరరాశి int64_t elems_చొప్పున_cl   = 16;
        స్థిరరాశి int64_t మొత్తం_cl       = (మొత్తం_elements + elems_చొప్పున_cl - 1) / elems_చొప్పున_cl;

        స్థిరరాశి int64_t cl_చొప్పున_thread = (మొత్తం_cl + అంకె_దారాలు - 1) / అంకె_దారాలు;
        స్థిరరాశి int64_t cl_ప్రారంభం      = thread_id * cl_చొప్పున_thread;
        int64_t       cl_చివర        = cl_ప్రారంభం + cl_చొప్పున_thread;
        ఒకవేళ (cl_చివర > మొత్తం_cl) {
            cl_చివర = మొత్తం_cl;
        }
        ఒకవేళ (cl_ప్రారంభం >= మొత్తం_cl) {
            తిరిగి 0;
        }

        స్థిరరాశి int64_t es = cl_ప్రారంభం * elems_చొప్పున_cl;
        int64_t       ee = cl_చివర * elems_చొప్పున_cl;
        ఒకవేళ (ee > మొత్తం_elements) {
            ee = మొత్తం_elements;
        }

        int64_t అడ్డువరుస_idx = es / ne0;
        int64_t నిలువు     = es % ne0;

        int64_t pos = es;
        అయితే (pos < ee) {
            స్థిరరాశి int64_t i1 = అడ్డువరుస_idx % ne1;
            స్థిరరాశి int64_t i2 = (అడ్డువరుస_idx / ne1) % ne2;
            స్థిరరాశి int64_t i3 = అడ్డువరుస_idx / (ne1 * ne2);

            స్థిరరాశి దశాంశం * src_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i1 * nb01 + i2 * nb02 + i3 * nb03);

            int64_t అడ్డువరుస_మిగిలిన   = ne0 - నిలువు;
            int64_t chunk_మిగిలిన = ee - pos;
            int64_t n               = అడ్డువరుస_మిగిలిన < chunk_మిగిలిన ? అడ్డువరుస_మిగిలిన : chunk_మిగిలిన;

            int64_t keep_ప్రారంభం, keep_చివర;
            ఎంపిక (tri_type) {
                స్థితి TRI_TYPE_క్రింది:
                    keep_ప్రారంభం = 0;
                    keep_చివర   = i1;
                    ఆపు;
                స్థితి TRI_TYPE_క్రింది_DIAG:
                    keep_ప్రారంభం = 0;
                    keep_చివర   = i1 + 1;
                    ఆపు;
                స్థితి TRI_TYPE_పై:
                    keep_ప్రారంభం = i1 + 1;
                    keep_చివర   = ne0;
                    ఆపు;
                స్థితి TRI_TYPE_పై_DIAG:
                    keep_ప్రారంభం = i1;
                    keep_చివర   = ne0;
                    ఆపు;
                అప్రమేయ:
                    తిరిగి -1;
            }
            ఒకవేళ (keep_చివర > ne0) {
                keep_చివర = ne0;
            }

            int64_t చివర_నిలువు = నిలువు + n;
            కొరకు (int64_t i0 = నిలువు; i0 < చివర_నిలువు; i0++) {
                ఒకవేళ (i0 >= keep_ప్రారంభం && i0 < keep_చివర) {
                    dst_దత్తాంశం[pos + (i0 - నిలువు)] = src_అడ్డువరుస[i0];
                } లేకపోతే {
                    dst_దత్తాంశం[pos + (i0 - నిలువు)] = 0.0f;
                }
            }

            pos += n;
            నిలువు = 0;
            అడ్డువరుస_idx++;
        }
    }

    తిరిగి 0;
}
