#pragma once

#অন্তর্ভাব "common.h"

#ifdef __cplusplus
বাহ্যিক "C" {
#endif

শূন্য ggml_compute_forward_abs(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_sgn(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_neg(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_step(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_tanh(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_elu(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_relu(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_sigmoid(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_hardsigmoid(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_exp(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_hardswish(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_sqr(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_sqrt(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_sin(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_cos(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_log(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_expm1(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_softplus(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_floor(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_ceil(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_round(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_trunc(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);
শূন্য ggml_compute_forward_xielu(ধ্রুবক কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
