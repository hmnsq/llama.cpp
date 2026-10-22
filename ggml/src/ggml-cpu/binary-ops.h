#pragma once

#அந்தர்பாவ "common.h"

#ifdef __cplusplus
வெளி "C" {
#endif

வெறுமை ggml_compute_forward_add_non_quantized(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_sub(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_mul(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);
வெறுமை ggml_compute_forward_div(மாறிலி கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
