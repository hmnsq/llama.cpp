#اگر_غیر_تعریف_شدہ GGML_ZDNN_UTILITIES_HPP
#تعریف_کرو GGML_ZDNN_UTILITIES_HPP

#شامل "common.hpp"

zdnn_ڈیٹا_types ggml_zdnn_type_mapping(ggml_type type);

خالی ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      ثابت ggml_tensor       * src,
                      ثابت int64_t           * ne,
                      ثابت zdnn_ڈیٹا_layouts   layout);

خالی ggml_zdnn_بھرو_tensor(zdnn_ztensor & ztensor, خالی * بفر);

int64_t ggml_zdnn_q_گروہ_سائز(ثابت ggml_tensor * tensor);

خالی ggml_zdnn_بھرو_tensor_q(ggml_backend_zdnn_بفر * بفر, ثابت ggml_tensor * tensor);

خالی ggml_zdnn_init_tensor(ggml_backend_zdnn_بفر * بفر, ثابت ggml_tensor * tensor);

#اگر_ختم  // GGML_ZDNN_UTILITIES_HPP
