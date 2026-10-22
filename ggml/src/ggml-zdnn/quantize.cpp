#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "quantize.hpp"

#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <cmath>

// quantize each group of group_size values per row to int8, plus the rounding error into residuals for ~15 bits
ശൂന്യം ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         സ്ഥിരം ggml_tensor * inputs,
                               int64_t       group_size,
                         സ്ഥിരം ദശാംശം       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               ദശാംശം       * scales) {

    സ്ഥിരം int64_t ne0      = inputs->ne[0];
    സ്ഥിരം int64_t ne1      = inputs->ne[1];
    സ്ഥിരം int64_t n_groups = (ne0 + group_size - 1) / group_size;

    ggml_zdnn_parallel_for(n_threads, ne1, [&](int64_t i1_start, int64_t i1_end) {
        വേണ്ടി (int64_t i1 = i1_start; i1 < i1_end; i1++) {
            സ്ഥിരം ദശാംശം * xi = (സ്ഥിരം ദശാംശം *)((സ്ഥിരം അക്ഷരം *)inputs->data + i1*inputs->nb[1]);

            വേണ്ടി (int64_t g = 0; g < n_groups; g++) {
                സ്ഥിരം ദശാംശം * xg = xi + g*group_size;
                സ്ഥിരം ദശാംശം * cg = channel_scales + g*group_size;
                സ്ഥിരം int64_t n  = std::min(group_size, ne0 - g*group_size);

                ദശാംശം amax = 0.0f;
                വേണ്ടി (int64_t i0 = 0; i0 < n; i0++) {
                    amax = std::max(amax, fabsf(xg[i0] * cg[i0]));
                }

                // the rounding error is at most d/2
                സ്ഥിരം ദശാംശം d      = amax / 127.0f;
                സ്ഥിരം ദശാംശം id     = d ? 1.0f/d : 0.0f;
                സ്ഥിരം ദശാംശം id_res = 254.0f * id;
                scales[g*ne1 + i1] = d;

                int8_t * qg = quants    + (g*ne1 + i1)*group_size;
                int8_t * rg = residuals + (g*ne1 + i1)*group_size;

                വേണ്ടി (int64_t i0 = 0; i0 < n; i0++) {
                    സ്ഥിരം ദശാംശം xc = xg[i0] * cg[i0];
                    സ്ഥിരം ദശാംശം v  = roundf(xc * id);

                    qg[i0] = (int8_t)v;
                    rg[i0] = (int8_t)std::clamp(roundf((xc - v * d) * id_res), -127.0f, 127.0f);
                }

                വേണ്ടി (int64_t i0 = n; i0 < group_size; i0++) {
                    qg[i0] = 0;
                    rg[i0] = 0;
                }
            }
        }
    });
}
