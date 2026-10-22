#अन्तर्भाव "ggml.h"
#अन्तर्भाव "quantize.hpp"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cmath>

// quantize each group of group_size values per row to int8, plus the rounding error into residuals for ~15 bits
शून्य ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         स्थिरांक ggml_tensor * inputs,
                               int64_t       group_size,
                         स्थिरांक दशांश       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               दशांश       * scales) {

    स्थिरांक int64_t ne0      = inputs->ne[0];
    स्थिरांक int64_t ne1      = inputs->ne[1];
    स्थिरांक int64_t n_groups = (ne0 + group_size - 1) / group_size;

    ggml_zdnn_parallel_for(n_threads, ne1, [&](int64_t i1_start, int64_t i1_end) {
        साठी (int64_t i1 = i1_start; i1 < i1_end; i1++) {
            स्थिरांक दशांश * xi = (स्थिरांक दशांश *)((स्थिरांक अक्षर *)inputs->data + i1*inputs->nb[1]);

            साठी (int64_t g = 0; g < n_groups; g++) {
                स्थिरांक दशांश * xg = xi + g*group_size;
                स्थिरांक दशांश * cg = channel_scales + g*group_size;
                स्थिरांक int64_t n  = std::min(group_size, ne0 - g*group_size);

                दशांश amax = 0.0f;
                साठी (int64_t i0 = 0; i0 < n; i0++) {
                    amax = std::max(amax, fabsf(xg[i0] * cg[i0]));
                }

                // the rounding error is at most d/2
                स्थिरांक दशांश d      = amax / 127.0f;
                स्थिरांक दशांश id     = d ? 1.0f/d : 0.0f;
                स्थिरांक दशांश id_res = 254.0f * id;
                scales[g*ne1 + i1] = d;

                int8_t * qg = quants    + (g*ne1 + i1)*group_size;
                int8_t * rg = residuals + (g*ne1 + i1)*group_size;

                साठी (int64_t i0 = 0; i0 < n; i0++) {
                    स्थिरांक दशांश xc = xg[i0] * cg[i0];
                    स्थिरांक दशांश v  = roundf(xc * id);

                    qg[i0] = (int8_t)v;
                    rg[i0] = (int8_t)std::clamp(roundf((xc - v * d) * id_res), -127.0f, 127.0f);
                }

                साठी (int64_t i0 = n; i0 < group_size; i0++) {
                    qg[i0] = 0;
                    rg[i0] = 0;
                }
            }
        }
    });
}
