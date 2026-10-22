#pragma once

#अन्तर्भाव "common.h"

#ifdef __cplusplus
बाह्य "C" {
#endif

शून्य ggml_compute_forward_add_non_quantized(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_sub(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_mul(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_div(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
