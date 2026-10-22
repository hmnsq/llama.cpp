#ifndef GGML_ZDNN_UTILITIES_HPP
#define GGML_ZDNN_UTILITIES_HPP

#अन्तर्भाव "common.hpp"

zdnn_data_types ggml_zdnn_type_mapping(ggml_type type);

शून्य ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      स्थिरांक ggml_tensor       * src,
                      स्थिरांक int64_t           * ne,
                      स्थिरांक zdnn_data_layouts   layout);

शून्य ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, शून्य * buffer);

int64_t ggml_zdnn_q_group_size(स्थिरांक ggml_tensor * tensor);

शून्य ggml_zdnn_load_tensor_q(ggml_backend_zdnn_buffer * buffer, स्थिरांक ggml_tensor * tensor);

शून्य ggml_zdnn_init_tensor(ggml_backend_zdnn_buffer * buffer, स्थिरांक ggml_tensor * tensor);

#endif  // GGML_ZDNN_UTILITIES_HPP
