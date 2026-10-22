#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "quantize.hpp"

#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <cmath>

// quantize each group of group_size values per row to int8, plus the rounding error into residuals for ~15 bits
শূন্য ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         ধ্রুবক ggml_tensor * inputs,
                               int64_t       group_আকার,
                         ধ্রুবক দশমিক       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               দশমিক       * scales) {

    ধ্রুবক int64_t ne0      = inputs->ne[0];
    ধ্রুবক int64_t ne1      = inputs->ne[1];
    ধ্রুবক int64_t n_groups = (ne0 + group_আকার - 1) / group_আকার;

    ggml_zdnn_parallel_for(n_threads, ne1, [&](int64_t i1_start, int64_t i1_end) {
        জন্য (int64_t i1 = i1_start; i1 < i1_end; i1++) {
            ধ্রুবক দশমিক * xi = (ধ্রুবক দশমিক *)((ধ্রুবক অক্ষর *)inputs->তথ্য + i1*inputs->nb[1]);

            জন্য (int64_t g = 0; g < n_groups; g++) {
                ধ্রুবক দশমিক * xg = xi + g*group_আকার;
                ধ্রুবক দশমিক * cg = channel_scales + g*group_আকার;
                ধ্রুবক int64_t n  = std::min(group_আকার, ne0 - g*group_আকার);

                দশমিক amax = 0.0f;
                জন্য (int64_t i0 = 0; i0 < n; i0++) {
                    amax = std::max(amax, fabsf(xg[i0] * cg[i0]));
                }

                // the rounding error is at most d/2
                ধ্রুবক দশমিক d      = amax / 127.0f;
                ধ্রুবক দশমিক id     = d ? 1.0f/d : 0.0f;
                ধ্রুবক দশমিক id_res = 254.0f * id;
                scales[g*ne1 + i1] = d;

                int8_t * qg = quants    + (g*ne1 + i1)*group_আকার;
                int8_t * rg = residuals + (g*ne1 + i1)*group_আকার;

                জন্য (int64_t i0 = 0; i0 < n; i0++) {
                    ধ্রুবক দশমিক xc = xg[i0] * cg[i0];
                    ধ্রুবক দশমিক v  = roundf(xc * id);

                    qg[i0] = (int8_t)v;
                    rg[i0] = (int8_t)std::clamp(roundf((xc - v * d) * id_res), -127.0f, 127.0f);
                }

                জন্য (int64_t i0 = n; i0 < group_আকার; i0++) {
                    qg[i0] = 0;
                    rg[i0] = 0;
                }
            }
        }
    });
}
