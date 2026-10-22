#pragma once

#अन्तर्भाव "common.h"

#ifdef __cplusplus
बाह्य "C" {
#endif

शून्य ggml_compute_forward_add_non_quantized(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_sub(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_mul(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_div(ध्रुव संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
