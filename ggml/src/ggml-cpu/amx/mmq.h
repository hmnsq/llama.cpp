#pragma once
#ଅନ୍ତର୍ଭାଵ "common.h"

size_t ggml_backend_amx_desired_wsize(ଅଚଳ ସଂରଚନା ggml_tensor * dst);

size_t ggml_backend_amx_get_alloc_size(ଅଚଳ ସଂରଚନା ggml_tensor * tensor);

ଶୂନ୍ୟ ggml_backend_amx_convert_weight(ସଂରଚନା ggml_tensor * tensor, ଅଚଳ ଶୂନ୍ୟ * data, size_t offset, size_t size);

ଶୂନ୍ୟ ggml_backend_amx_mul_mat(ଅଚଳ ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * dst);
