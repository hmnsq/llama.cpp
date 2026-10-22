#आशय once
#समावेश "common.h"

माप_प्रकार ggml_backend_amx_desired_wsize(स्थिर काष्ठा ggml_tensor * dst);

माप_प्रकार ggml_backend_amx_get_alloc_आकार(स्थिर काष्ठा ggml_tensor * tensor);

व्योम ggml_backend_amx_convert_weight(काष्ठा ggml_tensor * tensor, स्थिर व्योम * तथ्य, माप_प्रकार offset, माप_प्रकार आकार);

व्योम ggml_backend_amx_mul_mat(स्थिर काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * dst);
