#ifndef GGML_ZDNN_QUANTIZE_HPP
#define GGML_ZDNN_QUANTIZE_HPP

#ଅନ୍ତର୍ଭାଵ "common.hpp"

ଶୂନ୍ୟ ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         ଅଚଳ ggml_tensor * inputs,
                               int64_t       group_size,
                         ଅଚଳ ଦଶମିକ       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               ଦଶମିକ       * scales);

#endif  // GGML_ZDNN_QUANTIZE_HPP
