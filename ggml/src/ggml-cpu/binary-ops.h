#pragma once

#অন্তর্ভাৱ "common.h"

#ifdef __cplusplus
বাহ্য "C" {
#endif

শূন্য ggml_compute_forward_add_non_quantized(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_sub(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_mul(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_div(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
