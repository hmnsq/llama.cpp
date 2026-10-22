#pragma once

#ଅନ୍ତର୍ଭାଵ "common.h"

#ifdef __cplusplus
ବାହ୍ୟ "C" {
#endif

ଶୂନ୍ୟ ggml_compute_forward_add_non_quantized(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_sub(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_mul(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
ଶୂନ୍ୟ ggml_compute_forward_div(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
