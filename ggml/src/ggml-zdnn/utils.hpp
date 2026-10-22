#جے_بنا_تعریف GGML_ZDNN_UTILITIES_HPP
#تعریف_کرو GGML_ZDNN_UTILITIES_HPP

#ضمن "common.hpp"

zdnn_انکڑے_types ggml_zdnn_type_mapping(ggml_type type);

فراغ ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      ثابت ggml_tensor       * src,
                      ثابت int64_t           * ne,
                      ثابت zdnn_انکڑے_layouts   layout);

فراغ ggml_zdnn_لدو_tensor(zdnn_ztensor & ztensor, فراغ * بفر);

int64_t ggml_zdnn_q_گروہ_ناپ(ثابت ggml_tensor * tensor);

فراغ ggml_zdnn_لدو_tensor_q(ggml_backend_zdnn_بفر * بفر, ثابت ggml_tensor * tensor);

فراغ ggml_zdnn_init_tensor(ggml_backend_zdnn_بفر * بفر, ثابت ggml_tensor * tensor);

#جے_انت  // GGML_ZDNN_UTILITIES_HPP
