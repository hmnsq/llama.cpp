#pragma once

#অন্তর্ভাব "common.h"

#ifdef __cplusplus
বাহ্যিক "C" {
#endif

শূন্য ggml_compute_forward_add_non_quantized(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_sub(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_mul(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_div(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
