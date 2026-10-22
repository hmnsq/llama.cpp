#اگر_تعریف_نشده GGML_ZDNN_UTILITIES_HPP
#تعریف_کن GGML_ZDNN_UTILITIES_HPP

#ضمن "common.hpp"

zdnn_داده_types ggml_zdnn_type_mapping(ggml_type type);

فراغ ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      ثابت ggml_tensor       * src,
                      ثابت int64_t           * ne,
                      ثابت zdnn_داده_layouts   layout);

فراغ ggml_zdnn_بارگیری_tensor(zdnn_ztensor & ztensor, فراغ * میانگیر);

int64_t ggml_zdnn_q_گروه_اندازه(ثابت ggml_tensor * tensor);

فراغ ggml_zdnn_بارگیری_tensor_q(ggml_backend_zdnn_میانگیر * میانگیر, ثابت ggml_tensor * tensor);

فراغ ggml_zdnn_init_tensor(ggml_backend_zdnn_میانگیر * میانگیر, ثابت ggml_tensor * tensor);

#پایان_اگر  // GGML_ZDNN_UTILITIES_HPP
