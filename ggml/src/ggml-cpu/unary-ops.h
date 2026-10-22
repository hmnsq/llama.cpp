#pragma once

#অন্তর্ভাৱ "common.h"

#ifdef __cplusplus
বাহ্য "C" {
#endif

শূন্য ggml_compute_forward_abs(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_sgn(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_neg(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_step(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_tanh(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_elu(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_relu(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_sigmoid(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_hardsigmoid(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_exp(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_hardswish(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_sqr(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_sqrt(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_sin(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_cos(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_log(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_expm1(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_softplus(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_floor(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_ceil(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_round(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_trunc(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);
শূন্য ggml_compute_forward_xielu(ধ্রুৱ সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
