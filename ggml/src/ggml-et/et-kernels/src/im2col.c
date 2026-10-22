//******************************************************************************
// IM2COL Kernel
// Baseline scalar implementation for:
//   src1: [N, IC, IH, IW] -> dst: [N, OH, OW, IC*KH*KW] (2D)
//   src1: [N, IC,    IW] -> dst: [N,  1, OW, IC*   KW] (1D)
//
// Work is distributed by row-groups so threads own cache-line-aligned chunks of
// destination rows even when ne0 is not cache aligned.
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "math_fp.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>

స్థిర అంతఃవరుస శూన్య im2col_store_elem(శూన్య * dst_పునాది, గణన ggml_type dst_type, int64_t idx, దశాంశం విలువ) {
    ఒకవేళ (dst_type == GGML_TYPE_F32) {
        ((దశాంశం *) dst_పునాది)[idx] = విలువ;
    } లేకపోతే {
        ((uint16_t *) dst_పునాది)[idx] = fp32_గమ్యం_fp16(విలువ);
    }
}

స్థిర అంతఃవరుస దశాంశం im2col_ఎక్కించు_src_elem(స్థిరరాశి శూన్య * src_పునాది, గణన ggml_type src_type, int64_t idx) {
    ఒకవేళ (src_type == GGML_TYPE_F32) {
        తిరిగి ((స్థిరరాశి దశాంశం *) src_పునాది)[idx];
    }

    తిరిగి fp16_గమ్యం_fp32(((స్థిరరాశి uint16_t *) src_పునాది)[idx]);
}

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_ద్విమాన_params * params, శూన్య * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ఒకవేళ (!kernel_env || params == 0 || ((uint64_t) params & 0x7) != 0) {
        తిరిగి -1;
    }

    పూర్ణాంకం thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    పూర్ణాంకం అంకె_దారాలు = get_అంకె_దారాలు(kernel_env->shire_ముసుగు);

    ఒకవేళ (thread_id < 0) {
        తిరిగి 0;
    }

    నిర్మాణం ggml_tensor * src0 = &params->src0;
    నిర్మాణం ggml_tensor * src1 = &params->src1;
    నిర్మాణం ggml_tensor * dst  = &params->dst;

    ఒకవేళ (!src1->దత్తాంశం || !dst->దత్తాంశం) {
        తిరిగి -1;
    }

    ఒకవేళ (!((dst->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) ||
          (dst->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32)))) {
        తిరిగి -1;
    }

    స్థిరరాశి int32_t s0    = ((స్థిరరాశి int32_t *) dst->op_params)[0];
    స్థిరరాశి int32_t s1    = ((స్థిరరాశి int32_t *) dst->op_params)[1];
    స్థిరరాశి int32_t p0    = ((స్థిరరాశి int32_t *) dst->op_params)[2];
    స్థిరరాశి int32_t p1    = ((స్థిరరాశి int32_t *) dst->op_params)[3];
    స్థిరరాశి int32_t d0    = ((స్థిరరాశి int32_t *) dst->op_params)[4];
    స్థిరరాశి int32_t d1    = ((స్థిరరాశి int32_t *) dst->op_params)[5];
    స్థిరరాశి int32_t is_2d = ((స్థిరరాశి int32_t *) dst->op_params)[6];

    స్థిరరాశి int64_t N  = is_2d ? src1->ne[3] : src1->ne[2];
    స్థిరరాశి int64_t IC = is_2d ? src1->ne[2] : src1->ne[1];
    స్థిరరాశి int64_t IH = is_2d ? src1->ne[1] : 1;
    స్థిరరాశి int64_t IW = src1->ne[0];

    స్థిరరాశి int64_t KH = is_2d ? src0->ne[1] : 1;
    స్థిరరాశి int64_t KW = src0->ne[0];

    స్థిరరాశి int64_t OH         = is_2d ? dst->ne[2] : 1;
    స్థిరరాశి int64_t OW         = dst->ne[1];
    స్థిరరాశి int64_t అడ్డువరుస_elems  = dst->ne[0];
    స్థిరరాశి int64_t మొత్తం_అడ్డువరుసలు = OW * OH * N;

    స్థిరరాశి size_t src_batch_stride   = is_2d ? src1->nb[3] : src1->nb[2];
    స్థిరరాశి size_t src_channel_stride = is_2d ? src1->nb[2] : src1->nb[1];

    స్థిరరాశి size_t dst_అడ్డువరుస_stride   = dst->nb[1];
    స్థిరరాశి size_t dst_plane_stride = is_2d ? dst->nb[2] : 0;
    స్థిరరాశి size_t dst_batch_stride = is_2d ? dst->nb[3] : dst->nb[2];

    స్థిరరాశి int64_t dst_elem_కొలత  = (dst->type == GGML_TYPE_F32) ? (int64_t) పరిమాణం(దశాంశం) : (int64_t) పరిమాణం(uint16_t);
    స్థిరరాశి int64_t అడ్డువరుసలు_చొప్పున_సమూహం = et_అడ్డువరుసలు_చొప్పున_cacheline_సమూహం(అడ్డువరుస_elems, dst_elem_కొలత);
    స్థిరరాశి int64_t మొత్తం_groups   = (మొత్తం_అడ్డువరుసలు + అడ్డువరుసలు_చొప్పున_సమూహం - 1) / అడ్డువరుసలు_చొప్పున_సమూహం;

    కొరకు (int64_t grp = thread_id; grp < మొత్తం_groups; grp += అంకె_దారాలు) {
        స్థిరరాశి int64_t అడ్డువరుస_ప్రారంభం = grp * అడ్డువరుసలు_చొప్పున_సమూహం;
        int64_t       అడ్డువరుస_చివర   = అడ్డువరుస_ప్రారంభం + అడ్డువరుసలు_చొప్పున_సమూహం;
        ఒకవేళ (అడ్డువరుస_చివర > మొత్తం_అడ్డువరుసలు) {
            అడ్డువరుస_చివర = మొత్తం_అడ్డువరుసలు;
        }

        కొరకు (int64_t అడ్డువరుస = అడ్డువరుస_ప్రారంభం; అడ్డువరుస < అడ్డువరుస_చివర; ++అడ్డువరుస) {
            స్థిరరాశి int64_t లోపల  = అడ్డువరుస / (OH * OW);
            స్థిరరాశి int64_t rem = అడ్డువరుస % (OH * OW);
            స్థిరరాశి int64_t ioh = rem / OW;
            స్థిరరాశి int64_t iow = rem % OW;

            శూన్య * dst_అడ్డువరుస = (అక్షరం *) dst->దత్తాంశం + లోపల * dst_batch_stride + ioh * dst_plane_stride + iow * dst_అడ్డువరుస_stride;

            కొరకు (int64_t iic = 0; iic < IC; ++iic) {
                స్థిరరాశి శూన్య * src_channel = (స్థిరరాశి అక్షరం *) src1->దత్తాంశం + లోపల * src_batch_stride + iic * src_channel_stride;

                కొరకు (int64_t ikh = 0; ikh < KH; ++ikh) {
                    కొరకు (int64_t ikw = 0; ikw < KW; ++ikw) {
                        స్థిరరాశి int64_t iiw     = iow * s0 + ikw * d0 - p0;
                        స్థిరరాశి int64_t iih     = ioh * s1 + ikh * d1 - p1;
                        స్థిరరాశి int64_t dst_idx = iic * (KH * KW) + ikh * KW + ikw;

                        ఒకవేళ (iiw < 0 || iiw >= IW || iih < 0 || iih >= IH) {
                            im2col_store_elem(dst_అడ్డువరుస, dst->type, dst_idx, 0.0f);
                        } లేకపోతే {
                            స్థిరరాశి int64_t src_idx = iih * IW + iiw;
                            స్థిరరాశి దశాంశం   విలువ   = im2col_ఎక్కించు_src_elem(src_channel, src1->type, src_idx);
                            im2col_store_elem(dst_అడ్డువరుస, dst->type, dst_idx, విలువ);
                        }
                    }
                }
            }
        }
    }

    తిరిగి 0;
}
