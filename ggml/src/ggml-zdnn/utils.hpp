#جيڪڏهن_وصف_نه_ٿيل GGML_ZDNN_UTILITIES_HPP
#وصف_ڪريو GGML_ZDNN_UTILITIES_HPP

#ضمن "common.hpp"

zdnn_معلومات_types ggml_zdnn_type_mapping(ggml_type type);

فراغ ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      ثابت ggml_tensor       * src,
                      ثابت int64_t           * ne,
                      ثابت zdnn_معلومات_layouts   layout);

فراغ ggml_zdnn_ڀريو_tensor(zdnn_ztensor & ztensor, فراغ * بفر);

int64_t ggml_zdnn_q_گروهه_سائيز(ثابت ggml_tensor * tensor);

فراغ ggml_zdnn_ڀريو_tensor_q(ggml_backend_zdnn_بفر * بفر, ثابت ggml_tensor * tensor);

فراغ ggml_zdnn_init_tensor(ggml_backend_zdnn_بفر * بفر, ثابت ggml_tensor * tensor);

#جيڪڏهن_پورو  // GGML_ZDNN_UTILITIES_HPP
