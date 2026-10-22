#pragma once

#அந்தர்பாவ "common.h"

#ifdef __cplusplus
வெளி "C" {
#endif

வெறுமை ggml_compute_forward_abs(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_sgn(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_neg(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_step(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_tanh(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_elu(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_relu(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_sigmoid(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_hardsigmoid(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_exp(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_hardswish(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_sqr(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_sqrt(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_sin(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_cos(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_log(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_expm1(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_softplus(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_floor(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_ceil(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_round(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_trunc(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_xielu(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
