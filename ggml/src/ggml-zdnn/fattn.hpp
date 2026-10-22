#ifndef GGML_ZDNN_FATTN_HPP
#define GGML_ZDNN_FATTN_HPP

#அந்தர்பாவ "common.hpp"

தருக்க ggml_zdnn_supports_flash_attn_ext(மாறிலி ggml_backend_zdnn_device_context * ctx_dev, மாறிலி ggml_tensor * op);

வெறுமை ggml_zdnn_flash_attn_ext(ggml_backend_zdnn_context * ctx, ggml_tensor * dst);

#endif  // GGML_ZDNN_FATTN_HPP
