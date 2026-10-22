#pragma once
#અન્તર્ભાવ "common.h"

size_t ggml_backend_amx_desired_wsize(અચલ સંરચના ggml_tensor * dst);

size_t ggml_backend_amx_get_alloc_size(અચલ સંરચના ggml_tensor * tensor);

શૂન્ય ggml_backend_amx_convert_weight(સંરચના ggml_tensor * tensor, અચલ શૂન્ય * data, size_t offset, size_t size);

શૂન્ય ggml_backend_amx_mul_mat(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
