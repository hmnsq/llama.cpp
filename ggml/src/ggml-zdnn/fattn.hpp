#ifndef GGML_ZDNN_FATTN_HPP
#define GGML_ZDNN_FATTN_HPP

#شامل "common.hpp"

منطقی ggml_zdnn_supports_flash_attn_ext(ثابت ggml_backend_zdnn_device_context * ctx_dev, ثابت ggml_tensor * op);

تهی ggml_zdnn_flash_attn_ext(ggml_backend_zdnn_context * ctx, ggml_tensor * dst);

#endif  // GGML_ZDNN_FATTN_HPP
