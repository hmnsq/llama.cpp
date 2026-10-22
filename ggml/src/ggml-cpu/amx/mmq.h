#pragma once
#അന്തര്ഭാവ "common.h"

size_t ggml_backend_amx_desired_wsize(സ്ഥിരം ഘടന ggml_tensor * dst);

size_t ggml_backend_amx_get_alloc_size(സ്ഥിരം ഘടന ggml_tensor * tensor);

ശൂന്യം ggml_backend_amx_convert_weight(ഘടന ggml_tensor * tensor, സ്ഥിരം ശൂന്യം * data, size_t offset, size_t size);

ശൂന്യം ggml_backend_amx_mul_mat(സ്ഥിരം ഘടന ggml_compute_params * params, ഘടന ggml_tensor * dst);
