#ifndef GGML_ZDNN_FATTN_HPP
#define GGML_ZDNN_FATTN_HPP

#অন্তর্ভাব "common.hpp"

যৌক্তিক ggml_zdnn_supports_flash_attn_ext(ধ্রুবক ggml_backend_zdnn_device_context * ctx_dev, ধ্রুবক ggml_tensor * op);

শূন্য ggml_zdnn_flash_attn_ext(ggml_backend_zdnn_context * ctx, ggml_tensor * dst);

#endif  // GGML_ZDNN_FATTN_HPP
