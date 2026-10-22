#ifndef GGML_ZDNN_QUANTIZE_HPP
#define GGML_ZDNN_QUANTIZE_HPP

#অন্তর্ভাব "common.hpp"

শূন্য ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         ধ্রুব ggml_tensor * inputs,
                               int64_t       group_size,
                         ধ্রুব ভিন্ন       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               ভিন্ন       * scales);

#endif  // GGML_ZDNN_QUANTIZE_HPP
