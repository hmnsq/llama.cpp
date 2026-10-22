#pragma once

#شامل "common.h"

#ifdef __cplusplus
بیرونی "C" {
#endif

تهی ggml_compute_forward_abs(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_sgn(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_neg(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_step(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_tanh(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_elu(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_relu(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_sigmoid(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_hardsigmoid(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_exp(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_hardswish(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_sqr(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_sqrt(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_sin(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_cos(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_log(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_expm1(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_softplus(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_floor(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_ceil(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_round(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_trunc(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_xielu(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
