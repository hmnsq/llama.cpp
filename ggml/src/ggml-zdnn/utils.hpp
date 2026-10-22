#ifndef GGML_ZDNN_UTILITIES_HPP
#define GGML_ZDNN_UTILITIES_HPP

#ଅନ୍ତର୍ଭାଵ "common.hpp"

zdnn_data_types ggml_zdnn_type_mapping(ggml_type type);

ଶୂନ୍ୟ ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      ଅଚଳ ggml_tensor       * src,
                      ଅଚଳ int64_t           * ne,
                      ଅଚଳ zdnn_data_layouts   layout);

ଶୂନ୍ୟ ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, ଶୂନ୍ୟ * buffer);

int64_t ggml_zdnn_q_group_size(ଅଚଳ ggml_tensor * tensor);

ଶୂନ୍ୟ ggml_zdnn_load_tensor_q(ggml_backend_zdnn_buffer * buffer, ଅଚଳ ggml_tensor * tensor);

ଶୂନ୍ୟ ggml_zdnn_init_tensor(ggml_backend_zdnn_buffer * buffer, ଅଚଳ ggml_tensor * tensor);

#endif  // GGML_ZDNN_UTILITIES_HPP
