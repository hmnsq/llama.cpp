#ifndef GGML_ZDNN_QUANTIZE_HPP
#define GGML_ZDNN_QUANTIZE_HPP

#شامل "common.hpp"

تهی ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         ثابت ggml_tensor * inputs,
                               int64_t       group_size,
                         ثابت اعشاری       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               اعشاری       * scales);

#endif  // GGML_ZDNN_QUANTIZE_HPP
