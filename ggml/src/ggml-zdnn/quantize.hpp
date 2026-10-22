#ifndef GGML_ZDNN_QUANTIZE_HPP
#define GGML_ZDNN_QUANTIZE_HPP

#अन्तर्भाव "common.hpp"

शून्य ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         स्थिरांक ggml_tensor * inputs,
                               int64_t       group_size,
                         स्थिरांक दशांश       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               दशांश       * scales);

#endif  // GGML_ZDNN_QUANTIZE_HPP
