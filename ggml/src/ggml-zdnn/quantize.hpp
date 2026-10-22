#اگر_تعریف_نشده GGML_ZDNN_QUANTIZE_HPP
#تعریف_کن GGML_ZDNN_QUANTIZE_HPP

#ضمن "common.hpp"

فراغ ggml_zdnn_quantize_inputs(int32_t       n_نخها,
                         ثابت ggml_tensor * inputs,
                               int64_t       گروه_اندازه,
                         ثابت عائم       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               عائم       * scales);

#پایان_اگر  // GGML_ZDNN_QUANTIZE_HPP
