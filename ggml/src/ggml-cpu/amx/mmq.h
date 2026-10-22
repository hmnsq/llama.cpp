#pragma once
#अन्तर्भाव "common.h"

माप_प्रकार ggml_backend_amx_desired_wsize(ध्रुव संरचना ggml_tensor * dst);

माप_प्रकार ggml_backend_amx_get_alloc_size(ध्रुव संरचना ggml_tensor * tensor);

शून्य ggml_backend_amx_convert_weight(संरचना ggml_tensor * tensor, ध्रुव शून्य * data, माप_प्रकार offset, माप_प्रकार size);

शून्य ggml_backend_amx_mul_mat(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
