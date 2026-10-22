#অন্তর্ভাৱ "ggml.h"
#অন্তর্ভাৱ "quantize.hpp"

#অন্তর্ভাৱ <algorithm>
#অন্তর্ভাৱ <cmath>

// quantize each group of group_size values per row to int8, plus the rounding error into residuals for ~15 bits
শূন্য ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         ধ্রুৱ ggml_tensor * inputs,
                               int64_t       group_size,
                         ধ্রুৱ ভিন্ন       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               ভিন্ন       * scales) {

    ধ্রুৱ int64_t ne0      = inputs->ne[0];
    ধ্রুৱ int64_t ne1      = inputs->ne[1];
    ধ্রুৱ int64_t n_groups = (ne0 + group_size - 1) / group_size;

    ggml_zdnn_parallel_for(n_threads, ne1, [&](int64_t i1_start, int64_t i1_end) {
        বাবে (int64_t i1 = i1_start; i1 < i1_end; i1++) {
            ধ্রুৱ ভিন্ন * xi = (ধ্রুৱ ভিন্ন *)((ধ্রুৱ ৱর্ণ *)inputs->data + i1*inputs->nb[1]);

            বাবে (int64_t g = 0; g < n_groups; g++) {
                ধ্রুৱ ভিন্ন * xg = xi + g*group_size;
                ধ্রুৱ ভিন্ন * cg = channel_scales + g*group_size;
                ধ্রুৱ int64_t n  = std::min(group_size, ne0 - g*group_size);

                ভিন্ন amax = 0.0f;
                বাবে (int64_t i0 = 0; i0 < n; i0++) {
                    amax = std::max(amax, fabsf(xg[i0] * cg[i0]));
                }

                // the rounding error is at most d/2
                ধ্রুৱ ভিন্ন d      = amax / 127.0f;
                ধ্রুৱ ভিন্ন id     = d ? 1.0f/d : 0.0f;
                ধ্রুৱ ভিন্ন id_res = 254.0f * id;
                scales[g*ne1 + i1] = d;

                int8_t * qg = quants    + (g*ne1 + i1)*group_size;
                int8_t * rg = residuals + (g*ne1 + i1)*group_size;

                বাবে (int64_t i0 = 0; i0 < n; i0++) {
                    ধ্রুৱ ভিন্ন xc = xg[i0] * cg[i0];
                    ধ্রুৱ ভিন্ন v  = roundf(xc * id);

                    qg[i0] = (int8_t)v;
                    rg[i0] = (int8_t)std::clamp(roundf((xc - v * d) * id_res), -127.0f, 127.0f);
                }

                বাবে (int64_t i0 = n; i0 < group_size; i0++) {
                    qg[i0] = 0;
                    rg[i0] = 0;
                }
            }
        }
    });
}
