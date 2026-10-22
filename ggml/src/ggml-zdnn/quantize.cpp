#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "quantize.hpp"

#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <cmath>

// quantize each group of group_size values per row to int8, plus the rounding error into residuals for ~15 bits
வெறுமை ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         மாறிலி ggml_tensor * inputs,
                               int64_t       group_size,
                         மாறிலி தசம       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               தசம       * scales) {

    மாறிலி int64_t ne0      = inputs->ne[0];
    மாறிலி int64_t ne1      = inputs->ne[1];
    மாறிலி int64_t n_groups = (ne0 + group_size - 1) / group_size;

    ggml_zdnn_parallel_for(n_threads, ne1, [&](int64_t i1_start, int64_t i1_end) {
        க்கு (int64_t i1 = i1_start; i1 < i1_end; i1++) {
            மாறிலி தசம * xi = (மாறிலி தசம *)((மாறிலி எழுத்து *)inputs->data + i1*inputs->nb[1]);

            க்கு (int64_t g = 0; g < n_groups; g++) {
                மாறிலி தசம * xg = xi + g*group_size;
                மாறிலி தசம * cg = channel_scales + g*group_size;
                மாறிலி int64_t n  = std::min(group_size, ne0 - g*group_size);

                தசம amax = 0.0f;
                க்கு (int64_t i0 = 0; i0 < n; i0++) {
                    amax = std::max(amax, fabsf(xg[i0] * cg[i0]));
                }

                // the rounding error is at most d/2
                மாறிலி தசம d      = amax / 127.0f;
                மாறிலி தசம id     = d ? 1.0f/d : 0.0f;
                மாறிலி தசம id_res = 254.0f * id;
                scales[g*ne1 + i1] = d;

                int8_t * qg = quants    + (g*ne1 + i1)*group_size;
                int8_t * rg = residuals + (g*ne1 + i1)*group_size;

                க்கு (int64_t i0 = 0; i0 < n; i0++) {
                    மாறிலி தசம xc = xg[i0] * cg[i0];
                    மாறிலி தசம v  = roundf(xc * id);

                    qg[i0] = (int8_t)v;
                    rg[i0] = (int8_t)std::clamp(roundf((xc - v * d) * id_res), -127.0f, 127.0f);
                }

                க்கு (int64_t i0 = n; i0 < group_size; i0++) {
                    qg[i0] = 0;
                    rg[i0] = 0;
                }
            }
        }
    });
}
