#ifndef GGML_ZDNN_QUANTIZE_HPP
#define GGML_ZDNN_QUANTIZE_HPP

#অন্তর্ভাব "common.hpp"

শূন্য ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         ধ্রুবক ggml_tensor * inputs,
                               int64_t       group_আকার,
                         ধ্রুবক দশমিক       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               দশমিক       * scales);

#endif  // GGML_ZDNN_QUANTIZE_HPP
