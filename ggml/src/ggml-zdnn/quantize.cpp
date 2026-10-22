#شامل "ggml.h"
#شامل "quantize.hpp"

#شامل <algorithm>
#شامل <cmath>

// quantize each group of group_size values per row to int8, plus the rounding error into residuals for ~15 bits
خالی ggml_zdnn_quantize_inputs(int32_t       n_دھاگے,
                         ثابت ggml_tensor * inputs,
                               int64_t       گروہ_سائز,
                         ثابت اعشاری       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               اعشاری       * scales) {

    ثابت int64_t ne0      = inputs->ne[0];
    ثابت int64_t ne1      = inputs->ne[1];
    ثابت int64_t n_groups = (ne0 + گروہ_سائز - 1) / گروہ_سائز;

    ggml_zdnn_parallel_for(n_دھاگے, ne1, [&](int64_t i1_آغاز, int64_t i1_انتہا) {
        برائے (int64_t i1 = i1_آغاز; i1 < i1_انتہا; i1++) {
            ثابت اعشاری * xi = (ثابت اعشاری *)((ثابت حرف *)inputs->ڈیٹا + i1*inputs->nb[1]);

            برائے (int64_t g = 0; g < n_groups; g++) {
                ثابت اعشاری * xg = xi + g*گروہ_سائز;
                ثابت اعشاری * cg = channel_scales + g*گروہ_سائز;
                ثابت int64_t n  = std::min(گروہ_سائز, ne0 - g*گروہ_سائز);

                اعشاری amax = 0.0f;
                برائے (int64_t i0 = 0; i0 < n; i0++) {
                    amax = std::max(amax, fabsf(xg[i0] * cg[i0]));
                }

                // the rounding error is at most d/2
                ثابت اعشاری d      = amax / 127.0f;
                ثابت اعشاری id     = d ? 1.0f/d : 0.0f;
                ثابت اعشاری id_res = 254.0f * id;
                scales[g*ne1 + i1] = d;

                int8_t * qg = quants    + (g*ne1 + i1)*گروہ_سائز;
                int8_t * rg = residuals + (g*ne1 + i1)*گروہ_سائز;

                برائے (int64_t i0 = 0; i0 < n; i0++) {
                    ثابت اعشاری xc = xg[i0] * cg[i0];
                    ثابت اعشاری v  = roundf(xc * id);

                    qg[i0] = (int8_t)v;
                    rg[i0] = (int8_t)std::clamp(roundf((xc - v * d) * id_res), -127.0f, 127.0f);
                }

                برائے (int64_t i0 = n; i0 < گروہ_سائز; i0++) {
                    qg[i0] = 0;
                    rg[i0] = 0;
                }
            }
        }
    });
}
