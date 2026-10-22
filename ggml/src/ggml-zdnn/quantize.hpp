#ifndef GGML_ZDNN_QUANTIZE_HPP
#define GGML_ZDNN_QUANTIZE_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"

ಶೂನ್ಯ ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         ಧ್ರುವ ggml_tensor * inputs,
                               int64_t       group_size,
                         ಧ್ರುವ ಭಿನ್ನ       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               ಭಿನ್ನ       * scales);

#endif  // GGML_ZDNN_QUANTIZE_HPP
