#pragma once
#ಅನ್ತರ್ಭಾವ "common.h"

size_t ggml_backend_amx_desired_wsize(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * dst);

size_t ggml_backend_amx_get_alloc_size(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * tensor);

ಶೂನ್ಯ ggml_backend_amx_convert_weight(ಸಂರಚನಾ ggml_tensor * tensor, ಧ್ರುವ ಶೂನ್ಯ * data, size_t offset, size_t size);

ಶೂನ್ಯ ggml_backend_amx_mul_mat(ಧ್ರುವ ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * dst);
