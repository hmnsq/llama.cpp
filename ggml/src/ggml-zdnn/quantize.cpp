#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "quantize.hpp"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cmath>

// quantize each group of group_size values per row to int8, plus the rounding error into residuals for ~15 bits
ಶೂನ್ಯ ggml_zdnn_quantize_inputs(int32_t       n_ಎಳೆಗಳು,
                         ಸ್ಥಿರಾಂಕ ggml_tensor * inputs,
                               int64_t       ಗುಂಪು_ಗಾತ್ರ,
                         ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               ದಶಮಾಂಶ       * scales) {

    ಸ್ಥಿರಾಂಕ int64_t ne0      = inputs->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne1      = inputs->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_groups = (ne0 + ಗುಂಪು_ಗಾತ್ರ - 1) / ಗುಂಪು_ಗಾತ್ರ;

    ggml_zdnn_parallel_for(n_ಎಳೆಗಳು, ne1, [&](int64_t i1_ಆರಂಭ, int64_t i1_ಕೊನೆ) {
        ಗಾಗಿ (int64_t i1 = i1_ಆರಂಭ; i1 < i1_ಕೊನೆ; i1++) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * xi = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)inputs->ದತ್ತಾಂಶ + i1*inputs->nb[1]);

            ಗಾಗಿ (int64_t g = 0; g < n_groups; g++) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * xg = xi + g*ಗುಂಪು_ಗಾತ್ರ;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * cg = channel_scales + g*ಗುಂಪು_ಗಾತ್ರ;
                ಸ್ಥಿರಾಂಕ int64_t n  = std::min(ಗುಂಪು_ಗಾತ್ರ, ne0 - g*ಗುಂಪು_ಗಾತ್ರ);

                ದಶಮಾಂಶ amax = 0.0f;
                ಗಾಗಿ (int64_t i0 = 0; i0 < n; i0++) {
                    amax = std::max(amax, fabsf(xg[i0] * cg[i0]));
                }

                // the rounding error is at most d/2
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d      = amax / 127.0f;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id     = d ? 1.0f/d : 0.0f;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ id_res = 254.0f * id;
                scales[g*ne1 + i1] = d;

                int8_t * qg = quants    + (g*ne1 + i1)*ಗುಂಪು_ಗಾತ್ರ;
                int8_t * rg = residuals + (g*ne1 + i1)*ಗುಂಪು_ಗಾತ್ರ;

                ಗಾಗಿ (int64_t i0 = 0; i0 < n; i0++) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ xc = xg[i0] * cg[i0];
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v  = roundf(xc * id);

                    qg[i0] = (int8_t)v;
                    rg[i0] = (int8_t)std::clamp(roundf((xc - v * d) * id_res), -127.0f, 127.0f);
                }

                ಗಾಗಿ (int64_t i0 = n; i0 < ಗುಂಪು_ಗಾತ್ರ; i0++) {
                    qg[i0] = 0;
                    rg[i0] = 0;
                }
            }
        }
    });
}
