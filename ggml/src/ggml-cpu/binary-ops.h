#pragma once

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

void ggml_compute_forward_add_non_quantized(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_sub(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_mul(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);
void ggml_compute_forward_div(const sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
