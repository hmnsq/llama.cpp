#pragma once
#شامل "common.h"

size_t ggml_backend_amx_desired_wsize(ثابت ساختار ggml_tensor * dst);

size_t ggml_backend_amx_get_alloc_size(ثابت ساختار ggml_tensor * tensor);

تهی ggml_backend_amx_convert_weight(ساختار ggml_tensor * tensor, ثابت تهی * data, size_t offset, size_t size);

تهی ggml_backend_amx_mul_mat(ثابت ساختار ggml_compute_params * params, ساختار ggml_tensor * dst);
