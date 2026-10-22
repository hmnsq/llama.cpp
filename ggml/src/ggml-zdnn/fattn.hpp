#ifndef GGML_ZDNN_FATTN_HPP
#define GGML_ZDNN_FATTN_HPP

#अन्तर्भाव "common.hpp"

तार्किक ggml_zdnn_supports_flash_attn_ext(स्थिरांक ggml_backend_zdnn_device_context * ctx_dev, स्थिरांक ggml_tensor * op);

शून्य ggml_zdnn_flash_attn_ext(ggml_backend_zdnn_context * ctx, ggml_tensor * dst);

#endif  // GGML_ZDNN_FATTN_HPP
