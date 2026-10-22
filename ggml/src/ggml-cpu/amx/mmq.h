#pragma once
#ضمن "common.h"

size_t ggml_backend_amx_desired_wsize(ثابت بنية ggml_tensor * dst);

size_t ggml_backend_amx_get_alloc_size(ثابت بنية ggml_tensor * tensor);

فراغ ggml_backend_amx_convert_weight(بنية ggml_tensor * tensor, ثابت فراغ * data, size_t offset, size_t size);

فراغ ggml_backend_amx_mul_mat(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
