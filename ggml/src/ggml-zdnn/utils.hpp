#अगर_अघोषित GGML_ZDNN_UTILITIES_HPP
#घोषणा GGML_ZDNN_UTILITIES_HPP

#समावेश "common.hpp"

zdnn_तथ्य_types ggml_zdnn_type_mapping(ggml_type type);

व्योम ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      स्थिर ggml_tensor       * src,
                      स्थिर int64_t           * ne,
                      स्थिर zdnn_तथ्य_layouts   layout);

व्योम ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, व्योम * बफर);

int64_t ggml_zdnn_q_group_आकार(स्थिर ggml_tensor * tensor);

व्योम ggml_zdnn_load_tensor_q(ggml_backend_zdnn_बफर * बफर, स्थिर ggml_tensor * tensor);

व्योम ggml_zdnn_init_tensor(ggml_backend_zdnn_बफर * बफर, स्थिर ggml_tensor * tensor);

#पूर्ण_अगर  // GGML_ZDNN_UTILITIES_HPP
