#pragma once

#അന്തര്ഭാവ "common.h"

#ifdef __cplusplus
ബാഹ്യ "C" {
#endif

ശൂന്യം ggml_compute_forward_abs(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_sgn(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_neg(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_step(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_tanh(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_elu(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_relu(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_sigmoid(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_hardsigmoid(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_exp(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_hardswish(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_sqr(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_sqrt(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_sin(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_cos(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_log(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_expm1(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_softplus(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_floor(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_ceil(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_round(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_trunc(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
ശൂന്യം ggml_compute_forward_xielu(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
