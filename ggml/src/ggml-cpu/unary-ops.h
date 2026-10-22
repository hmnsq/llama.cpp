#pragma once

#અન્તર્ભાવ "common.h"

#ifdef __cplusplus
બાહ્ય "C" {
#endif

શૂન્ય ggml_compute_forward_abs(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_sgn(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_neg(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_step(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_tanh(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_elu(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_relu(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_sigmoid(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_hardsigmoid(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_exp(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_hardswish(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_sqr(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_sqrt(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_sin(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_cos(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_log(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_expm1(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_softplus(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_floor(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_ceil(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_round(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_trunc(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);
શૂન્ય ggml_compute_forward_xielu(અચલ સંરચના ggml_compute_params * params, સંરચના ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
