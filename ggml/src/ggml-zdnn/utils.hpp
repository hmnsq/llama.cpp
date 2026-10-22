#ifndef GGML_ZDNN_UTILITIES_HPP
#define GGML_ZDNN_UTILITIES_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"

zdnn_data_types ggml_zdnn_type_mapping(ggml_type type);

ಶೂನ್ಯ ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      ಧ್ರುವ ggml_tensor       * src,
                      ಧ್ರುವ int64_t           * ne,
                      ಧ್ರುವ zdnn_data_layouts   layout);

ಶೂನ್ಯ ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, ಶೂನ್ಯ * buffer);

int64_t ggml_zdnn_q_group_size(ಧ್ರುವ ggml_tensor * tensor);

ಶೂನ್ಯ ggml_zdnn_load_tensor_q(ggml_backend_zdnn_buffer * buffer, ಧ್ರುವ ggml_tensor * tensor);

ಶೂನ್ಯ ggml_zdnn_init_tensor(ggml_backend_zdnn_buffer * buffer, ಧ್ರುವ ggml_tensor * tensor);

#endif  // GGML_ZDNN_UTILITIES_HPP
