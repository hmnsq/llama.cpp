#ifndef GGML_ZDNN_QUANTIZE_HPP
#define GGML_ZDNN_QUANTIZE_HPP

#અન્તર્ભાવ "common.hpp"

શૂન્ય ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         અચલ ggml_tensor * inputs,
                               int64_t       group_size,
                         અચલ દશાંશ       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               દશાંશ       * scales);

#endif  // GGML_ZDNN_QUANTIZE_HPP
