#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "quantize.hpp"

#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <cmath>

// quantize each group of group_size values per row to int8, plus the rounding error into residuals for ~15 bits
શૂન્ય ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         અચલ ggml_tensor * inputs,
                               int64_t       group_size,
                         અચલ દશાંશ       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               દશાંશ       * scales) {

    અચલ int64_t ne0      = inputs->ne[0];
    અચલ int64_t ne1      = inputs->ne[1];
    અચલ int64_t n_groups = (ne0 + group_size - 1) / group_size;

    ggml_zdnn_parallel_for(n_threads, ne1, [&](int64_t i1_start, int64_t i1_end) {
        માટે (int64_t i1 = i1_start; i1 < i1_end; i1++) {
            અચલ દશાંશ * xi = (અચલ દશાંશ *)((અચલ અક્ષર *)inputs->data + i1*inputs->nb[1]);

            માટે (int64_t g = 0; g < n_groups; g++) {
                અચલ દશાંશ * xg = xi + g*group_size;
                અચલ દશાંશ * cg = channel_scales + g*group_size;
                અચલ int64_t n  = std::min(group_size, ne0 - g*group_size);

                દશાંશ amax = 0.0f;
                માટે (int64_t i0 = 0; i0 < n; i0++) {
                    amax = std::max(amax, fabsf(xg[i0] * cg[i0]));
                }

                // the rounding error is at most d/2
                અચલ દશાંશ d      = amax / 127.0f;
                અચલ દશાંશ id     = d ? 1.0f/d : 0.0f;
                અચલ દશાંશ id_res = 254.0f * id;
                scales[g*ne1 + i1] = d;

                int8_t * qg = quants    + (g*ne1 + i1)*group_size;
                int8_t * rg = residuals + (g*ne1 + i1)*group_size;

                માટે (int64_t i0 = 0; i0 < n; i0++) {
                    અચલ દશાંશ xc = xg[i0] * cg[i0];
                    અચલ દશાંશ v  = roundf(xc * id);

                    qg[i0] = (int8_t)v;
                    rg[i0] = (int8_t)std::clamp(roundf((xc - v * d) * id_res), -127.0f, 127.0f);
                }

                માટે (int64_t i0 = n; i0 < group_size; i0++) {
                    qg[i0] = 0;
                    rg[i0] = 0;
                }
            }
        }
    });
}
