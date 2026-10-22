#ifndef GGML_ZDNN_UTILITIES_HPP
#define GGML_ZDNN_UTILITIES_HPP

#അന്തര്ഭാവ "common.hpp"

zdnn_data_types ggml_zdnn_type_mapping(ggml_type type);

ശൂന്യം ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      സ്ഥിരം ggml_tensor       * src,
                      സ്ഥിരം int64_t           * ne,
                      സ്ഥിരം zdnn_data_layouts   layout);

ശൂന്യം ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, ശൂന്യം * buffer);

int64_t ggml_zdnn_q_group_size(സ്ഥിരം ggml_tensor * tensor);

ശൂന്യം ggml_zdnn_load_tensor_q(ggml_backend_zdnn_buffer * buffer, സ്ഥിരം ggml_tensor * tensor);

ശൂന്യം ggml_zdnn_init_tensor(ggml_backend_zdnn_buffer * buffer, സ്ഥിരം ggml_tensor * tensor);

#endif  // GGML_ZDNN_UTILITIES_HPP
