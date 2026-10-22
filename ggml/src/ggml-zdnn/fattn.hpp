#ifndef GGML_ZDNN_FATTN_HPP
#define GGML_ZDNN_FATTN_HPP

#അന്തര്ഭാവ "common.hpp"

തർക്കം ggml_zdnn_supports_flash_attn_ext(സ്ഥിരം ggml_backend_zdnn_device_context * ctx_dev, സ്ഥിരം ggml_tensor * op);

ശൂന്യം ggml_zdnn_flash_attn_ext(ggml_backend_zdnn_context * ctx, ggml_tensor * dst);

#endif  // GGML_ZDNN_FATTN_HPP
