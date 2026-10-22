#pragma once

#അന്തര്ഭാവ "common.h"

#ifdef __cplusplus
ബാഹ്യ "C" {
#endif

ശൂന്യം ggml_compute_forward_add_non_quantized(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_sub(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_mul(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_div(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
