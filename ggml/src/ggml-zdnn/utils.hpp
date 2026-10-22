#ifndef GGML_ZDNN_UTILITIES_HPP
#define GGML_ZDNN_UTILITIES_HPP

#ضمن "common.hpp"

zdnn_data_types ggml_zdnn_type_mapping(ggml_type type);

فراغ ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      ثابت ggml_tensor       * src,
                      ثابت int64_t           * ne,
                      ثابت zdnn_data_layouts   layout);

فراغ ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, فراغ * buffer);

int64_t ggml_zdnn_q_group_size(ثابت ggml_tensor * tensor);

فراغ ggml_zdnn_load_tensor_q(ggml_backend_zdnn_buffer * buffer, ثابت ggml_tensor * tensor);

فراغ ggml_zdnn_init_tensor(ggml_backend_zdnn_buffer * buffer, ثابت ggml_tensor * tensor);

#endif  // GGML_ZDNN_UTILITIES_HPP
