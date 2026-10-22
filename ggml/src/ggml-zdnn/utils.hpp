#ifndef GGML_ZDNN_UTILITIES_HPP
#define GGML_ZDNN_UTILITIES_HPP

#અન્તર્ભાવ "common.hpp"

zdnn_data_types ggml_zdnn_type_mapping(ggml_type type);

શૂન્ય ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      અચલ ggml_tensor       * src,
                      અચલ int64_t           * ne,
                      અચલ zdnn_data_layouts   layout);

શૂન્ય ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, શૂન્ય * buffer);

int64_t ggml_zdnn_q_group_size(અચલ ggml_tensor * tensor);

શૂન્ય ggml_zdnn_load_tensor_q(ggml_backend_zdnn_buffer * buffer, અચલ ggml_tensor * tensor);

શૂન્ય ggml_zdnn_init_tensor(ggml_backend_zdnn_buffer * buffer, અચલ ggml_tensor * tensor);

#endif  // GGML_ZDNN_UTILITIES_HPP
