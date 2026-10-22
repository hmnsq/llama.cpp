#pragma once
#அந்தர்பாவ "common.h"

size_t ggml_backend_amx_desired_wsize(மாறிலி கட்டமைப்பு ggml_tensor * dst);

size_t ggml_backend_amx_get_alloc_size(மாறிலி கட்டமைப்பு ggml_tensor * tensor);

வெறுமை ggml_backend_amx_convert_weight(கட்டமைப்பு ggml_tensor * tensor, மாறிலி வெறுமை * data, size_t offset, size_t size);

வெறுமை ggml_backend_amx_mul_mat(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
