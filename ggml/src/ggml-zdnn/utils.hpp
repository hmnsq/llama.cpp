#ifndef GGML_ZDNN_UTILITIES_HPP
#define GGML_ZDNN_UTILITIES_HPP

#شامل "common.hpp"

zdnn_data_types ggml_zdnn_type_mapping(ggml_type type);

تهی ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      ثابت ggml_tensor       * src,
                      ثابت int64_t           * ne,
                      ثابت zdnn_data_layouts   layout);

تهی ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, تهی * buffer);

int64_t ggml_zdnn_q_group_size(ثابت ggml_tensor * tensor);

تهی ggml_zdnn_load_tensor_q(ggml_backend_zdnn_buffer * buffer, ثابت ggml_tensor * tensor);

تهی ggml_zdnn_init_tensor(ggml_backend_zdnn_buffer * buffer, ثابت ggml_tensor * tensor);

#endif  // GGML_ZDNN_UTILITIES_HPP
