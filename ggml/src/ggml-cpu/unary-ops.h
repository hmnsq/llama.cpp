#pragma once

#অন্তর্ভাব "common.h"

#ifdef __cplusplus
বাহ্য "C" {
#endif

শূন্য ggml_compute_forward_abs(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_sgn(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_neg(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_step(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_tanh(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_elu(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_relu(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_sigmoid(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_hardsigmoid(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_exp(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_hardswish(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_sqr(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_sqrt(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_sin(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_cos(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_log(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_expm1(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_softplus(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_floor(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_ceil(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_round(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_trunc(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_xielu(ধ্রুব সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
