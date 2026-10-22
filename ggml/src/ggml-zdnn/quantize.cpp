#ضمن "ggml.h"
#ضمن "quantize.hpp"

#ضمن <algorithm>
#ضمن <cmath>

// quantize each group of group_size values per row to int8, plus the rounding error into residuals for ~15 bits
فراغ ggml_zdnn_quantize_inputs(int32_t       n_ڌاڳا,
                         ثابت ggml_tensor * inputs,
                               int64_t       گروهه_سائيز,
                         ثابت عائم       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               عائم       * scales) {

    ثابت int64_t ne0      = inputs->ne[0];
    ثابت int64_t ne1      = inputs->ne[1];
    ثابت int64_t n_groups = (ne0 + گروهه_سائيز - 1) / گروهه_سائيز;

    ggml_zdnn_parallel_for(n_ڌاڳا, ne1, [&](int64_t i1_شروعات, int64_t i1_پڇاڙي) {
        لكل (int64_t i1 = i1_شروعات; i1 < i1_پڇاڙي; i1++) {
            ثابت عائم * xi = (ثابت عائم *)((ثابت حرف *)inputs->معلومات + i1*inputs->nb[1]);

            لكل (int64_t g = 0; g < n_groups; g++) {
                ثابت عائم * xg = xi + g*گروهه_سائيز;
                ثابت عائم * cg = channel_scales + g*گروهه_سائيز;
                ثابت int64_t n  = std::min(گروهه_سائيز, ne0 - g*گروهه_سائيز);

                عائم amax = 0.0f;
                لكل (int64_t i0 = 0; i0 < n; i0++) {
                    amax = std::max(amax, fabsf(xg[i0] * cg[i0]));
                }

                // the rounding error is at most d/2
                ثابت عائم d      = amax / 127.0f;
                ثابت عائم id     = d ? 1.0f/d : 0.0f;
                ثابت عائم id_res = 254.0f * id;
                scales[g*ne1 + i1] = d;

                int8_t * qg = quants    + (g*ne1 + i1)*گروهه_سائيز;
                int8_t * rg = residuals + (g*ne1 + i1)*گروهه_سائيز;

                لكل (int64_t i0 = 0; i0 < n; i0++) {
                    ثابت عائم xc = xg[i0] * cg[i0];
                    ثابت عائم v  = roundf(xc * id);

                    qg[i0] = (int8_t)v;
                    rg[i0] = (int8_t)std::clamp(roundf((xc - v * d) * id_res), -127.0f, 127.0f);
                }

                لكل (int64_t i0 = n; i0 < گروهه_سائيز; i0++) {
                    qg[i0] = 0;
                    rg[i0] = 0;
                }
            }
        }
    });
}
