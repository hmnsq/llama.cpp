#pragma once

#ضمن "common.h"

#ifdef __cplusplus
خارجي "C" {
#endif

فراغ ggml_compute_forward_add_non_quantized(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_sub(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_mul(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_div(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
