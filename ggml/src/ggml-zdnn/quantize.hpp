#ifndef GGML_ZDNN_QUANTIZE_HPP
#define GGML_ZDNN_QUANTIZE_HPP

#ضمن "common.hpp"

فراغ ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         ثابت ggml_tensor * inputs,
                               int64_t       group_size,
                         ثابت عائم       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               عائم       * scales);

#endif  // GGML_ZDNN_QUANTIZE_HPP
