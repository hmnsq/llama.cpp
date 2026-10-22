#यदि_अपरिभाषितम् GGML_ZDNN_UTILITIES_HPP
#परिभाषय GGML_ZDNN_UTILITIES_HPP

#अन्तर्भाव "common.hpp"

zdnn_दत्त_types ggml_zdnn_type_mapping(ggml_type type);

शून्य ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      ध्रुव ggml_tensor       * src,
                      ध्रुव int64_t           * ne,
                      ध्रुव zdnn_दत्त_layouts   layout);

शून्य ggml_zdnn_आरोपय_tensor(zdnn_ztensor & ztensor, शून्य * बफर);

int64_t ggml_zdnn_q_गणः_आकार(ध्रुव ggml_tensor * tensor);

शून्य ggml_zdnn_आरोपय_tensor_q(ggml_backend_zdnn_बफर * बफर, ध्रुव ggml_tensor * tensor);

शून्य ggml_zdnn_init_tensor(ggml_backend_zdnn_बफर * बफर, ध्रुव ggml_tensor * tensor);

#यदि_अन्तः  // GGML_ZDNN_UTILITIES_HPP
