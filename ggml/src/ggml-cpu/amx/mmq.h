#هدايت once
#ضمن "common.h"

size_t ggml_backend_amx_desired_wsize(ثابت بنية ggml_tensor * dst);

size_t ggml_backend_amx_get_alloc_سائيز(ثابت بنية ggml_tensor * tensor);

فراغ ggml_backend_amx_convert_وزن(بنية ggml_tensor * tensor, ثابت فراغ * معلومات, size_t وٿي, size_t سائيز);

فراغ ggml_backend_amx_mul_mat(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
