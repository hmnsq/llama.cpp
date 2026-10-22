#అన్తర్భావ "ggml.h"
#అన్తర్భావ "quantize.hpp"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cmath>

// quantize each group of group_size values per row to int8, plus the rounding error into residuals for ~15 bits
శూన్య ggml_zdnn_quantize_inputs(int32_t       n_దారాలు,
                         స్థిరరాశి ggml_tensor * inputs,
                               int64_t       సమూహం_కొలత,
                         స్థిరరాశి దశాంశం       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               దశాంశం       * scales) {

    స్థిరరాశి int64_t ne0      = inputs->ne[0];
    స్థిరరాశి int64_t ne1      = inputs->ne[1];
    స్థిరరాశి int64_t n_groups = (ne0 + సమూహం_కొలత - 1) / సమూహం_కొలత;

    ggml_zdnn_parallel_for(n_దారాలు, ne1, [&](int64_t i1_ప్రారంభం, int64_t i1_చివర) {
        కొరకు (int64_t i1 = i1_ప్రారంభం; i1 < i1_చివర; i1++) {
            స్థిరరాశి దశాంశం * xi = (స్థిరరాశి దశాంశం *)((స్థిరరాశి అక్షరం *)inputs->దత్తాంశం + i1*inputs->nb[1]);

            కొరకు (int64_t g = 0; g < n_groups; g++) {
                స్థిరరాశి దశాంశం * xg = xi + g*సమూహం_కొలత;
                స్థిరరాశి దశాంశం * cg = channel_scales + g*సమూహం_కొలత;
                స్థిరరాశి int64_t n  = std::min(సమూహం_కొలత, ne0 - g*సమూహం_కొలత);

                దశాంశం amax = 0.0f;
                కొరకు (int64_t i0 = 0; i0 < n; i0++) {
                    amax = std::max(amax, fabsf(xg[i0] * cg[i0]));
                }

                // the rounding error is at most d/2
                స్థిరరాశి దశాంశం d      = amax / 127.0f;
                స్థిరరాశి దశాంశం id     = d ? 1.0f/d : 0.0f;
                స్థిరరాశి దశాంశం id_res = 254.0f * id;
                scales[g*ne1 + i1] = d;

                int8_t * qg = quants    + (g*ne1 + i1)*సమూహం_కొలత;
                int8_t * rg = residuals + (g*ne1 + i1)*సమూహం_కొలత;

                కొరకు (int64_t i0 = 0; i0 < n; i0++) {
                    స్థిరరాశి దశాంశం xc = xg[i0] * cg[i0];
                    స్థిరరాశి దశాంశం v  = roundf(xc * id);

                    qg[i0] = (int8_t)v;
                    rg[i0] = (int8_t)std::clamp(roundf((xc - v * d) * id_res), -127.0f, 127.0f);
                }

                కొరకు (int64_t i0 = n; i0 < సమూహం_కొలత; i0++) {
                    qg[i0] = 0;
                    rg[i0] = 0;
                }
            }
        }
    });
}
