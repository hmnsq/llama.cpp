#pragma once

#अन्तर्भाव "common.h"

#ifdef __cplusplus
बाह्य "C" {
#endif

शून्य ggml_compute_forward_abs(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_sgn(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_neg(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_step(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_tanh(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_elu(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_relu(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_sigmoid(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_hardsigmoid(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_exp(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_hardswish(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_sqr(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_sqrt(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_sin(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_cos(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_log(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_expm1(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_softplus(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_floor(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_ceil(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_round(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_trunc(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);
शून्य ggml_compute_forward_xielu(स्थिरांक संरचना ggml_compute_params * params, संरचना ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
