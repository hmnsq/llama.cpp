//******************************************************************************
// GROUP_NORM F32 Kernel
// Baseline scalar implementation:
//   normalize over (ne0 * ne1 * channels_in_group) for each (group, batch).
//
// Parallelization:
// - Work is partitioned across (group, batch) pairs.
// - For non-cache-aligned ne0, writes are emitted in row-groups so each thread's
//   destination write footprint still spans an integer number of cache lines.
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "math_fp.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>

నిర్మాణం ggml_et_సమూహం_norm_params {
    నిర్మాణం ggml_tensor src0;
    నిర్మాణం ggml_tensor dst;
    int32_t            n_groups;
    దశాంశం              eps;
};

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_సమూహం_norm_params * params, శూన్య * env) {
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

    స్థిరరాశి int32_t n_groups = params->n_groups;
    స్థిరరాశి దశాంశం   eps      = params->eps;

    ఒకవేళ (n_groups <= 0 || eps < 0.0f) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t ne0 = dst->ne[0];
    స్థిరరాశి int64_t ne1 = dst->ne[1];
    స్థిరరాశి int64_t ne2 = dst->ne[2];
    స్థిరరాశి int64_t ne3 = dst->ne[3];

    ఒకవేళ (src0->ne[0] != ne0 || src0->ne[1] != ne1 || src0->ne[2] != ne2 || src0->ne[3] != ne3) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t nb1  = dst->nb[1];
    స్థిరరాశి int64_t nb2  = dst->nb[2];
    స్థిరరాశి int64_t nb3  = dst->nb[3];
    స్థిరరాశి int64_t nb01 = src0->nb[1];
    స్థిరరాశి int64_t nb02 = src0->nb[2];
    స్థిరరాశి int64_t nb03 = src0->nb[3];

    స్థిరరాశి int64_t channels_చొప్పున_సమూహం = (ne2 + n_groups - 1) / n_groups;
    ఒకవేళ (channels_చొప్పున_సమూహం <= 0) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t చురుకైన_groups        = (ne2 + channels_చొప్పున_సమూహం - 1) / channels_చొప్పున_సమూహం;
    స్థిరరాశి int64_t మొత్తం_work           = చురుకైన_groups * ne3;
    స్థిరరాశి int64_t అడ్డువరుసలు_చొప్పున_write_సమూహం = et_అడ్డువరుసలు_చొప్పున_cacheline_సమూహం(ne0, పరిమాణం(దశాంశం));

    కొరకు (int64_t work = thread_id; work < మొత్తం_work; work += అంకె_దారాలు) {
        స్థిరరాశి int64_t i3        = work / చురుకైన_groups;
        స్థిరరాశి int64_t సమూహం_idx = work % చురుకైన_groups;

        స్థిరరాశి int64_t channel_ప్రారంభం = సమూహం_idx * channels_చొప్పున_సమూహం;
        int64_t       channel_చివర   = channel_ప్రారంభం + channels_చొప్పున_సమూహం;
        ఒకవేళ (channel_చివర > ne2) {
            channel_చివర = ne2;
        }

        స్థిరరాశి int64_t channel_లెక్క = channel_చివర - channel_ప్రారంభం;
        ఒకవేళ (channel_లెక్క <= 0) {
            కొనసాగు;
        }

        దశాంశం కూడిక   = 0.0f;
        దశాంశం denom = 0.0f;
        కొరకు (int64_t i2 = channel_ప్రారంభం; i2 < channel_చివర; ++i2) {
            కొరకు (int64_t i1 = 0; i1 < ne1; ++i1) {
                స్థిరరాశి దశాంశం * src_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i3 * nb03 + i2 * nb02 + i1 * nb01);
                కొరకు (int64_t i0 = 0; i0 < ne0; ++i0) {
                    కూడిక += src_అడ్డువరుస[i0];
                    denom += 1.0f;
                }
            }
        }

        స్థిరరాశి దశాంశం సగటు_2 = et_fdiv(కూడిక, denom);

        దశాంశం var_కూడిక = 0.0f;
        కొరకు (int64_t i2 = channel_ప్రారంభం; i2 < channel_చివర; ++i2) {
            కొరకు (int64_t i1 = 0; i1 < ne1; ++i1) {
                స్థిరరాశి దశాంశం * src_అడ్డువరుస = (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i3 * nb03 + i2 * nb02 + i1 * nb01);
                కొరకు (int64_t i0 = 0; i0 < ne0; ++i0) {
                    స్థిరరాశి దశాంశం centered = src_అడ్డువరుస[i0] - సగటు_2;
                    var_కూడిక += centered * centered;
                }
            }
        }

        స్థిరరాశి దశాంశం variance = et_fdiv(var_కూడిక, denom);
        స్థిరరాశి దశాంశం scale    = et_fdiv(1.0f, et_sqrtf(variance + eps));

        ఒకవేళ (ne0 % 16 == 0) {
            కొరకు (int64_t i2 = channel_ప్రారంభం; i2 < channel_చివర; ++i2) {
                కొరకు (int64_t i1 = 0; i1 < ne1; ++i1) {
                    స్థిరరాశి దశాంశం * src_అడ్డువరుస =
                        (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    దశాంశం * dst_అడ్డువరుస = (దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    కొరకు (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_అడ్డువరుస[i0] = (src_అడ్డువరుస[i0] - సగటు_2) * scale;
                    }
                }
            }
        } లేకపోతే {
            స్థిరరాశి int64_t మొత్తం_అడ్డువరుసలు_లోపల_సమూహం = channel_లెక్క * ne1;
            స్థిరరాశి int64_t మొత్తం_write_groups  = (మొత్తం_అడ్డువరుసలు_లోపల_సమూహం + అడ్డువరుసలు_చొప్పున_write_సమూహం - 1) / అడ్డువరుసలు_చొప్పున_write_సమూహం;

            కొరకు (int64_t write_సమూహం = 0; write_సమూహం < మొత్తం_write_groups; ++write_సమూహం) {
                స్థిరరాశి int64_t అడ్డువరుస_ప్రారంభం = write_సమూహం * అడ్డువరుసలు_చొప్పున_write_సమూహం;
                int64_t       అడ్డువరుస_చివర   = అడ్డువరుస_ప్రారంభం + అడ్డువరుసలు_చొప్పున_write_సమూహం;
                ఒకవేళ (అడ్డువరుస_చివర > మొత్తం_అడ్డువరుసలు_లోపల_సమూహం) {
                    అడ్డువరుస_చివర = మొత్తం_అడ్డువరుసలు_లోపల_సమూహం;
                }

                కొరకు (int64_t అడ్డువరుస = అడ్డువరుస_ప్రారంభం; అడ్డువరుస < అడ్డువరుస_చివర; ++అడ్డువరుస) {
                    స్థిరరాశి int64_t local_i2 = అడ్డువరుస / ne1;
                    స్థిరరాశి int64_t i1       = అడ్డువరుస % ne1;
                    స్థిరరాశి int64_t i2       = channel_ప్రారంభం + local_i2;

                    స్థిరరాశి దశాంశం * src_అడ్డువరుస =
                        (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + i3 * nb03 + i2 * nb02 + i1 * nb01);
                    దశాంశం * dst_అడ్డువరుస = (దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + i3 * nb3 + i2 * nb2 + i1 * nb1);
                    కొరకు (int64_t i0 = 0; i0 < ne0; ++i0) {
                        dst_అడ్డువరుస[i0] = (src_అడ్డువరుస[i0] - సగటు_2) * scale;
                    }
                }
            }
        }
    }

    తిరిగి 0;
}
