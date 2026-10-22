#pragma once
#অন্তর্ভাব "common.h"

size_t ggml_backend_amx_desired_wsize(ধ্রুব সংরচনা ggml_tensor * dst);

size_t ggml_backend_amx_get_alloc_size(ধ্রুব সংরচনা ggml_tensor * tensor);

শূন্য ggml_backend_amx_convert_weight(সংরচনা ggml_tensor * tensor, ধ্রুব শূন্য * data, size_t offset, size_t size);

শূন্য ggml_backend_amx_mul_mat(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
