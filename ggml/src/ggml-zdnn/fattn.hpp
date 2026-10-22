#ifndef GGML_ZDNN_FATTN_HPP
#define GGML_ZDNN_FATTN_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"

ತಾರ್ಕಿಕ ggml_zdnn_supports_flash_attn_ext(ಧ್ರುವ ggml_backend_zdnn_device_context * ctx_dev, ಧ್ರುವ ggml_tensor * op);

ಶೂನ್ಯ ggml_zdnn_flash_attn_ext(ggml_backend_zdnn_context * ctx, ggml_tensor * dst);

#endif  // GGML_ZDNN_FATTN_HPP
