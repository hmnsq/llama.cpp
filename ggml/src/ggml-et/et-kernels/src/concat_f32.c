//******************************************************************************
// Concat F32 Kernel
// Concatenates two F32 tensors along a specified dimension.
// All copies are aligned to cacheline boundaries (64 bytes = 16 floats).
//
// For dim >= 1, entire rows are copied from src0 or src1 into dst.
// For dim == 0, use:
// - a fast vector path when both source row segments are cacheline-aligned
// - a scalar stride-aware path otherwise
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>
#అన్తర్భావ <string.h>

నిర్మాణం ggml_et_concat_params {
    నిర్మాణం ggml_tensor src0;  // F32 input tensor 0
    నిర్మాణం ggml_tensor src1;  // F32 input tensor 1
    నిర్మాణం ggml_tensor dst;   // F32 output tensor
    int32_t            dim;   // Concatenation dimension
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

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_concat_params * params, శూన్య * env) {
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
    int32_t              dim  = params->dim;

    ఒకవేళ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి -1;
    }

    దశాంశం * src0_దత్తాంశం = (దశాంశం *) src0->దత్తాంశం;
    దశాంశం * src1_దత్తాంశం = (దశాంశం *) src1->దత్తాంశం;
    దశాంశం * dst_దత్తాంశం  = (దశాంశం *) dst->దత్తాంశం;

    ఒకవేళ (!src0_దత్తాంశం || !src1_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t ne00 = src0->ne[0], ne01 = src0->ne[1], ne02 = src0->ne[2], ne03 = src0->ne[3];
    స్థిరరాశి int64_t ne10 = src1->ne[0], ne11 = src1->ne[1], ne12 = src1->ne[2], ne13 = src1->ne[3];
    స్థిరరాశి int64_t ne0 = dst->ne[0], ne1 = dst->ne[1], ne2 = dst->ne[2], ne3 = dst->ne[3];

    // src strides in bytes
    స్థిరరాశి size_t nb00 = src0->nb[0], nb01 = src0->nb[1], nb02 = src0->nb[2], nb03 = src0->nb[3];
    స్థిరరాశి size_t nb10 = src1->nb[0], nb11 = src1->nb[1], nb12 = src1->nb[2], nb13 = src1->nb[3];
    // dst strides in bytes
    స్థిరరాశి size_t dnb1 = dst->nb[1], dnb2 = dst->nb[2], dnb3 = dst->nb[3];

    // Total rows across all higher dimensions
    స్థిరరాశి int64_t మొత్తం_అడ్డువరుసలు = ne1 * ne2 * ne3;

    // Generic slow path for dim==0 when either source segment is not suitable for
    // aligned vector copies. Threading is done by cacheline-aligned row groups,
    // so writers do not share destination cache lines.
    ఒకవేళ (dim == 0 && (ne00 % 16 != 0 || ne10 % 16 != 0 || nb00 != పరిమాణం(దశాంశం) || nb10 != పరిమాణం(దశాంశం))) {
        స్థిరరాశి int64_t అడ్డువరుసలు_చొప్పున_సమూహం = et_అడ్డువరుసలు_చొప్పున_cacheline_సమూహం(ne0, పరిమాణం(దశాంశం));
        స్థిరరాశి int64_t మొత్తం_groups   = (మొత్తం_అడ్డువరుసలు + అడ్డువరుసలు_చొప్పున_సమూహం - 1) / అడ్డువరుసలు_చొప్పున_సమూహం;

        కొరకు (int64_t grp = thread_id; grp < మొత్తం_groups; grp += అంకె_దారాలు) {
            స్థిరరాశి int64_t అడ్డువరుస_ప్రారంభం = grp * అడ్డువరుసలు_చొప్పున_సమూహం;
            int64_t       అడ్డువరుస_చివర   = అడ్డువరుస_ప్రారంభం + అడ్డువరుసలు_చొప్పున_సమూహం;
            ఒకవేళ (అడ్డువరుస_చివర > మొత్తం_అడ్డువరుసలు) {
                అడ్డువరుస_చివర = మొత్తం_అడ్డువరుసలు;
            }

            కొరకు (int64_t అడ్డువరుస = అడ్డువరుస_ప్రారంభం; అడ్డువరుస < అడ్డువరుస_చివర; అడ్డువరుస++) {
                int64_t i1 = అడ్డువరుస % ne1;
                int64_t i2 = (అడ్డువరుస / ne1) % ne2;
                int64_t i3 = అడ్డువరుస / (ne1 * ne2);

                దశాంశం * dst_అడ్డువరుస = (దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

                స్థిరరాశి అక్షరం * s0_పునాది = (స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i1 * nb01 + i2 * nb02 + i3 * nb03;
                కొరకు (int64_t i0 = 0; i0 < ne00; i0++) {
                    dst_అడ్డువరుస[i0] = *(స్థిరరాశి దశాంశం *) (s0_పునాది + i0 * nb00);
                }

                స్థిరరాశి అక్షరం * s1_పునాది = (స్థిరరాశి అక్షరం *) src1_దత్తాంశం + i1 * nb11 + i2 * nb12 + i3 * nb13;
                కొరకు (int64_t i0 = 0; i0 < ne10; i0++) {
                    dst_అడ్డువరుస[ne00 + i0] = *(స్థిరరాశి దశాంశం *) (s1_పునాది + i0 * nb10);
                }
            }
        }
        తిరిగి 0;
    }

    // Standard path: ne0 % 16 == 0, aligned rows
    కొరకు (int64_t అడ్డువరుస = thread_id; అడ్డువరుస < మొత్తం_అడ్డువరుసలు; అడ్డువరుస += అంకె_దారాలు) {
        // Decompose linear row index into (i1, i2, i3)
        int64_t i1 = అడ్డువరుస % ne1;
        int64_t i2 = (అడ్డువరుస / ne1) % ne2;
        int64_t i3 = అడ్డువరుస / (ne1 * ne2);

        దశాంశం * dst_అడ్డువరుస = (దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + i1 * dnb1 + i2 * dnb2 + i3 * dnb3);

        ఒకవేళ (dim == 0) {
            // Concat along innermost dimension: [src0_row | src1_row]
            // Both ne00 and ne10 are multiples of 16 (cacheline-aligned)
            స్థిరరాశి దశాంశం * s0_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i1 * nb01 + i2 * nb02 + i3 * nb03);
            స్థిరరాశి దశాంశం * s1_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1_దత్తాంశం + i1 * nb11 + i2 * nb12 + i3 * nb13);

            copy_అడ్డువరుస_aligned(dst_అడ్డువరుస, s0_అడ్డువరుస, (int32_t) ne00);
            copy_అడ్డువరుస_aligned(dst_అడ్డువరుస + ne00, s1_అడ్డువరుస, (int32_t) ne10);

        } లేకపోతే ఒకవేళ (dim == 1) {
            // Concat along dim 1: first ne01 rows from src0, rest from src1
            ఒకవేళ (i1 < ne01) {
                స్థిరరాశి దశాంశం * s0_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_అడ్డువరుస_aligned(dst_అడ్డువరుస, s0_అడ్డువరుస, (int32_t) ne0);
            } లేకపోతే {
                స్థిరరాశి దశాంశం * s1_అడ్డువరుస =
                    (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1_దత్తాంశం + (i1 - ne01) * nb11 + i2 * nb12 + i3 * nb13);
                copy_అడ్డువరుస_aligned(dst_అడ్డువరుస, s1_అడ్డువరుస, (int32_t) ne0);
            }

        } లేకపోతే ఒకవేళ (dim == 2) {
            // Concat along dim 2: first ne02 slices from src0, rest from src1
            ఒకవేళ (i2 < ne02) {
                స్థిరరాశి దశాంశం * s0_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_అడ్డువరుస_aligned(dst_అడ్డువరుస, s0_అడ్డువరుస, (int32_t) ne0);
            } లేకపోతే {
                స్థిరరాశి దశాంశం * s1_అడ్డువరుస =
                    (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1_దత్తాంశం + i1 * nb11 + (i2 - ne02) * nb12 + i3 * nb13);
                copy_అడ్డువరుస_aligned(dst_అడ్డువరుస, s1_అడ్డువరుస, (int32_t) ne0);
            }

        } లేకపోతే {
            // dim == 3: first ne03 batches from src0, rest from src1
            ఒకవేళ (i3 < ne03) {
                స్థిరరాశి దశాంశం * s0_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i1 * nb01 + i2 * nb02 + i3 * nb03);
                copy_అడ్డువరుస_aligned(dst_అడ్డువరుస, s0_అడ్డువరుస, (int32_t) ne0);
            } లేకపోతే {
                స్థిరరాశి దశాంశం * s1_అడ్డువరుస =
                    (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src1_దత్తాంశం + i1 * nb11 + i2 * nb12 + (i3 - ne03) * nb13);
                copy_అడ్డువరుస_aligned(dst_అడ్డువరుస, s1_అడ్డువరుస, (int32_t) ne0);
            }
        }
    }

    తిరిగి 0;
}
