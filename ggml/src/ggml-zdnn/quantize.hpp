#ifndef GGML_ZDNN_QUANTIZE_HPP
#define GGML_ZDNN_QUANTIZE_HPP

#அந்தர்பாவ "common.hpp"

வெறுமை ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         மாறிலி ggml_tensor * inputs,
                               int64_t       group_size,
                         மாறிலி தசம       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               தசம       * scales);

#endif  // GGML_ZDNN_QUANTIZE_HPP
