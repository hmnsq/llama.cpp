#pragma once
#অন্তর্ভাব "common.h"

size_t ggml_backend_amx_desired_wsize(ধ্রুবক কাঠামো ggml_tensor * dst);

size_t ggml_backend_amx_get_alloc_আকার(ধ্রুবক কাঠামো ggml_tensor * tensor);

শূন্য ggml_backend_amx_convert_weight(কাঠামো ggml_tensor * tensor, ধ্রুবক শূন্য * তথ্য, size_t offset, size_t আকার_2);

শূন্য ggml_backend_amx_mul_mat(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
