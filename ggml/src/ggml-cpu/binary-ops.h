#pragma once

#અન્તર્ભાવ "common.h"

#ifdef __cplusplus
બાહ્ય "C" {
#endif

શૂન્ય ggml_compute_forward_add_non_quantized(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_sub(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_mul(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_div(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
