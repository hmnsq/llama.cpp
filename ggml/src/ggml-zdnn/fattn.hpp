#ifndef GGML_ZDNN_FATTN_HPP
#define GGML_ZDNN_FATTN_HPP

#અન્તર્ભાવ "common.hpp"

તાર્કિક ggml_zdnn_supports_flash_attn_ext(અચલ ggml_backend_zdnn_device_context * ctx_dev, અચલ ggml_tensor * op);

શૂન્ય ggml_zdnn_flash_attn_ext(ggml_backend_zdnn_context * ctx, ggml_tensor * dst);

#endif  // GGML_ZDNN_FATTN_HPP
