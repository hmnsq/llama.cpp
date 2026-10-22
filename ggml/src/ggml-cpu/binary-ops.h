#رهنمود once

#ضمن "common.h"

#اگر_تعریف_شده __cplusplus
خارجي "C" {
#پایان_اگر

فراغ ggml_compute_forward_جمع_کن_non_quantized(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_sub(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_mul(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);
فراغ ggml_compute_forward_div(ثابت بنية ggml_compute_params * params, بنية ggml_tensor * dst);

#اگر_تعریف_شده __cplusplus
}
#پایان_اگر
