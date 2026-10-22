#ifndef GGML_ZDNN_UTILITIES_HPP
#define GGML_ZDNN_UTILITIES_HPP

#अन्तर्भाव "common.hpp"

zdnn_data_types ggml_zdnn_type_mapping(ggml_type type);

शून्य ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      ध्रुव ggml_tensor       * src,
                      ध्रुव int64_t           * ne,
                      ध्रुव zdnn_data_layouts   layout);

शून्य ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, शून्य * buffer);

int64_t ggml_zdnn_q_group_size(ध्रुव ggml_tensor * tensor);

शून्य ggml_zdnn_load_tensor_q(ggml_backend_zdnn_buffer * buffer, ध्रुव ggml_tensor * tensor);

शून्य ggml_zdnn_init_tensor(ggml_backend_zdnn_buffer * buffer, ध्रुव ggml_tensor * tensor);

#endif  // GGML_ZDNN_UTILITIES_HPP
