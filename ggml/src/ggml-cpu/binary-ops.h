#pragma once

#ಅನ್ತರ್ಭಾವ "common.h"

#ifdef __cplusplus
ಬಾಹ್ಯ "C" {
#endif

ಶೂನ್ಯ ggml_compute_forward_add_non_quantized(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_sub(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_mul(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
ಶೂನ್ಯ ggml_compute_forward_div(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
