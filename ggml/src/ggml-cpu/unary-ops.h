#pragma once

#ଅନ୍ତର୍ଭାଵ "common.h"

#ifdef __cplusplus
ବାହ୍ୟ "C" {
#endif

ଶୂନ୍ୟ ggml_compute_forward_abs(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_sgn(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_neg(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_step(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_tanh(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_elu(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_relu(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_sigmoid(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_hardsigmoid(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_exp(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_hardswish(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_sqr(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_sqrt(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_sin(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_cos(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_log(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_expm1(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_softplus(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_floor(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_ceil(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_round(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_trunc(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_xielu(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
