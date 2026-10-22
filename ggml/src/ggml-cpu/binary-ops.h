#pragma once

#شامل "common.h"

#ifdef __cplusplus
بیرونی "C" {
#endif

تهی ggml_compute_forward_add_non_quantized(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_sub(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_mul(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
تهی ggml_compute_forward_div(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
