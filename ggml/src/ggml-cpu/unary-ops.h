#pragma once

#ಅನ್ತರ್ಭಾವ "common.h"

#ifdef __cplusplus
ಬಾಹ್ಯ "C" {
#endif

ಶೂನ್ಯ ggml_compute_forward_abs(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_sgn(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_neg(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_step(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_tanh(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_elu(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_relu(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_sigmoid(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_hardsigmoid(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_exp(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_hardswish(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_sqr(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_sqrt(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_sin(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_cos(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_log(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_expm1(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_softplus(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_floor(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_ceil(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_round(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_trunc(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_xielu(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
