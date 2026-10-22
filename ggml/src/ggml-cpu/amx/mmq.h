#pragma once
#अन्तर्भाव "common.h"

माप_प्रकार ggml_backend_amx_desired_wsize(स्थिरांक संरचना ggml_tensor * dst);

माप_प्रकार ggml_backend_amx_get_alloc_size(स्थिरांक संरचना ggml_tensor * tensor);

शून्य ggml_backend_amx_convert_weight(संरचना ggml_tensor * tensor, स्थिरांक शून्य * data, माप_प्रकार offset, माप_प्रकार size);

शून्य ggml_backend_amx_mul_mat(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
