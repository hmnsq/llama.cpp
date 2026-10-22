#समावेश "ggml.h"
#समावेश "quantize.hpp"

#समावेश <algorithm>
#समावेश <cmath>

// quantize each group of group_size values per row to int8, plus the rounding error into residuals for ~15 bits
व्योम ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         स्थिर ggml_tensor * inputs,
                               int64_t       group_आकार,
                         स्थिर भग्न       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               भग्न       * scales) {

    स्थिर int64_t ne0      = inputs->ne[0];
    स्थिर int64_t ne1      = inputs->ne[1];
    स्थिर int64_t n_groups = (ne0 + group_आकार - 1) / group_आकार;

    ggml_zdnn_parallel_for(n_threads, ne1, [&](int64_t i1_start, int64_t i1_end) {
        क्रम (int64_t i1 = i1_start; i1 < i1_end; i1++) {
            स्थिर भग्न * xi = (स्थिर भग्न *)((स्थिर अक्षर *)inputs->तथ्य + i1*inputs->nb[1]);

            क्रम (int64_t g = 0; g < n_groups; g++) {
                स्थिर भग्न * xg = xi + g*group_आकार;
                स्थिर भग्न * cg = channel_scales + g*group_आकार;
                स्थिर int64_t n  = std::min(group_आकार, ne0 - g*group_आकार);

                भग्न amax = 0.0f;
                क्रम (int64_t i0 = 0; i0 < n; i0++) {
                    amax = std::max(amax, fabsf(xg[i0] * cg[i0]));
                }

                // the rounding error is at most d/2
                स्थिर भग्न d      = amax / 127.0f;
                स्थिर भग्न id     = d ? 1.0f/d : 0.0f;
                स्थिर भग्न id_res = 254.0f * id;
                scales[g*ne1 + i1] = d;

                int8_t * qg = quants    + (g*ne1 + i1)*group_आकार;
                int8_t * rg = residuals + (g*ne1 + i1)*group_आकार;

                क्रम (int64_t i0 = 0; i0 < n; i0++) {
                    स्थिर भग्न xc = xg[i0] * cg[i0];
                    स्थिर भग्न v  = roundf(xc * id);

                    qg[i0] = (int8_t)v;
                    rg[i0] = (int8_t)std::clamp(roundf((xc - v * d) * id_res), -127.0f, 127.0f);
                }

                क्रम (int64_t i0 = n; i0 < group_आकार; i0++) {
                    qg[i0] = 0;
                    rg[i0] = 0;
                }
            }
        }
    });
}
