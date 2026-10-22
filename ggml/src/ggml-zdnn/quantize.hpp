#ifndef GGML_ZDNN_QUANTIZE_HPP
#define GGML_ZDNN_QUANTIZE_HPP

#അന്തര്ഭാവ "common.hpp"

ശൂന്യം ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         സ്ഥിരം ggml_tensor * inputs,
                               int64_t       group_size,
                         സ്ഥിരം ദശാംശം       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               ദശാംശം       * scales);

#endif  // GGML_ZDNN_QUANTIZE_HPP
