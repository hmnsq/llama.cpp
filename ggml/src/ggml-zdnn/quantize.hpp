#که_نه_تعریف_شوی GGML_ZDNN_QUANTIZE_HPP
#تعریف_کړه GGML_ZDNN_QUANTIZE_HPP

#ضمن "common.hpp"

فراغ ggml_zdnn_quantize_inputs(int32_t       n_تارونه,
                         ثابت ggml_tensor * inputs,
                               int64_t       ډله_اندازه,
                         ثابت عائم       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               عائم       * scales);

#د_که_پای  // GGML_ZDNN_QUANTIZE_HPP
