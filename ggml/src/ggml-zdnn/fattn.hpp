#جے_بنا_تعریف GGML_ZDNN_FATTN_HPP
#تعریف_کرو GGML_ZDNN_FATTN_HPP

#ضمن "common.hpp"

منطقي ggml_zdnn_supports_flash_attn_ext(ثابت ggml_backend_zdnn_device_context * ctx_dev, ثابت ggml_tensor * op);

فراغ ggml_zdnn_flash_attn_ext(ggml_backend_zdnn_context * ctx, ggml_tensor * dst);

#جے_انت  // GGML_ZDNN_FATTN_HPP
