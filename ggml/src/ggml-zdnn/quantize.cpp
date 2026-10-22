#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "quantize.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>

// quantize each group of group_size values per row to int8, plus the rounding error into residuals for ~15 bits
ਖਾਲੀ ggml_zdnn_quantize_inputs(int32_t       n_ਧਾਗੇ,
                         ਅਚਲ ggml_tensor * inputs,
                               int64_t       ਸਮੂਹ_ਮਾਪ,
                         ਅਚਲ ਦਸ਼ਮਲਵ       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               ਦਸ਼ਮਲਵ       * scales) {

    ਅਚਲ int64_t ne0      = inputs->ne[0];
    ਅਚਲ int64_t ne1      = inputs->ne[1];
    ਅਚਲ int64_t n_groups = (ne0 + ਸਮੂਹ_ਮਾਪ - 1) / ਸਮੂਹ_ਮਾਪ;

    ggml_zdnn_parallel_for(n_ਧਾਗੇ, ne1, [&](int64_t i1_ਸ਼ੁਰੂ, int64_t i1_ਸਿਰਾ) {
        ਲਈ (int64_t i1 = i1_ਸ਼ੁਰੂ; i1 < i1_ਸਿਰਾ; i1++) {
            ਅਚਲ ਦਸ਼ਮਲਵ * xi = (ਅਚਲ ਦਸ਼ਮਲਵ *)((ਅਚਲ ਅੱਖਰ *)inputs->ਅੰਕੜੇ + i1*inputs->nb[1]);

            ਲਈ (int64_t g = 0; g < n_groups; g++) {
                ਅਚਲ ਦਸ਼ਮਲਵ * xg = xi + g*ਸਮੂਹ_ਮਾਪ;
                ਅਚਲ ਦਸ਼ਮਲਵ * cg = channel_scales + g*ਸਮੂਹ_ਮਾਪ;
                ਅਚਲ int64_t n  = std::min(ਸਮੂਹ_ਮਾਪ, ne0 - g*ਸਮੂਹ_ਮਾਪ);

                ਦਸ਼ਮਲਵ amax = 0.0f;
                ਲਈ (int64_t i0 = 0; i0 < n; i0++) {
                    amax = std::max(amax, fabsf(xg[i0] * cg[i0]));
                }

                // the rounding error is at most d/2
                ਅਚਲ ਦਸ਼ਮਲਵ d      = amax / 127.0f;
                ਅਚਲ ਦਸ਼ਮਲਵ id     = d ? 1.0f/d : 0.0f;
                ਅਚਲ ਦਸ਼ਮਲਵ id_res = 254.0f * id;
                scales[g*ne1 + i1] = d;

                int8_t * qg = quants    + (g*ne1 + i1)*ਸਮੂਹ_ਮਾਪ;
                int8_t * rg = residuals + (g*ne1 + i1)*ਸਮੂਹ_ਮਾਪ;

                ਲਈ (int64_t i0 = 0; i0 < n; i0++) {
                    ਅਚਲ ਦਸ਼ਮਲਵ xc = xg[i0] * cg[i0];
                    ਅਚਲ ਦਸ਼ਮਲਵ v  = roundf(xc * id);

                    qg[i0] = (int8_t)v;
                    rg[i0] = (int8_t)std::clamp(roundf((xc - v * d) * id_res), -127.0f, 127.0f);
                }

                ਲਈ (int64_t i0 = n; i0 < ਸਮੂਹ_ਮਾਪ; i0++) {
                    qg[i0] = 0;
                    rg[i0] = 0;
                }
            }
        }
    });
}
