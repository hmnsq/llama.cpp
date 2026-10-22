#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "quantize.hpp"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cmath>

// quantize each group of group_size values per row to int8, plus the rounding error into residuals for ~15 bits
ಶೂನ್ಯ ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         ಧ್ರುವ ggml_tensor * inputs,
                               int64_t       group_size,
                         ಧ್ರುವ ಭಿನ್ನ       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               ಭಿನ್ನ       * scales) {

    ಧ್ರುವ int64_t ne0      = inputs->ne[0];
    ಧ್ರುವ int64_t ne1      = inputs->ne[1];
    ಧ್ರುವ int64_t n_groups = (ne0 + group_size - 1) / group_size;

    ggml_zdnn_parallel_for(n_threads, ne1, [&](int64_t i1_start, int64_t i1_end) {
        ಪ್ರತಿ (int64_t i1 = i1_start; i1 < i1_end; i1++) {
            ಧ್ರುವ ಭಿನ್ನ * xi = (ಧ್ರುವ ಭಿನ್ನ *)((ಧ್ರುವ ವರ್ಣ *)inputs->data + i1*inputs->nb[1]);

            ಪ್ರತಿ (int64_t g = 0; g < n_groups; g++) {
                ಧ್ರುವ ಭಿನ್ನ * xg = xi + g*group_size;
                ಧ್ರುವ ಭಿನ್ನ * cg = channel_scales + g*group_size;
                ಧ್ರುವ int64_t n  = std::min(group_size, ne0 - g*group_size);

                ಭಿನ್ನ amax = 0.0f;
                ಪ್ರತಿ (int64_t i0 = 0; i0 < n; i0++) {
                    amax = std::max(amax, fabsf(xg[i0] * cg[i0]));
                }

                // the rounding error is at most d/2
                ಧ್ರುವ ಭಿನ್ನ d      = amax / 127.0f;
                ಧ್ರುವ ಭಿನ್ನ id     = d ? 1.0f/d : 0.0f;
                ಧ್ರುವ ಭಿನ್ನ id_res = 254.0f * id;
                scales[g*ne1 + i1] = d;

                int8_t * qg = quants    + (g*ne1 + i1)*group_size;
                int8_t * rg = residuals + (g*ne1 + i1)*group_size;

                ಪ್ರತಿ (int64_t i0 = 0; i0 < n; i0++) {
                    ಧ್ರುವ ಭಿನ್ನ xc = xg[i0] * cg[i0];
                    ಧ್ರುವ ಭಿನ್ನ v  = roundf(xc * id);

                    qg[i0] = (int8_t)v;
                    rg[i0] = (int8_t)std::clamp(roundf((xc - v * d) * id_res), -127.0f, 127.0f);
                }

                ಪ್ರತಿ (int64_t i0 = n; i0 < group_size; i0++) {
                    qg[i0] = 0;
                    rg[i0] = 0;
                }
            }
        }
    });
}
