#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_ZDNN_UTILITIES_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_ZDNN_UTILITIES_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"

zdnn_ਅੰਕੜੇ_types ggml_zdnn_type_mapping(ggml_type type);

ਖਾਲੀ ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      ਅਚਲ ggml_tensor       * src,
                      ਅਚਲ int64_t           * ne,
                      ਅਚਲ zdnn_ਅੰਕੜੇ_layouts   layout);

ਖਾਲੀ ggml_zdnn_ਲੱਦੋ_tensor(zdnn_ztensor & ztensor, ਖਾਲੀ * ਬਫਰ);

int64_t ggml_zdnn_q_ਸਮੂਹ_ਮਾਪ(ਅਚਲ ggml_tensor * tensor);

ਖਾਲੀ ggml_zdnn_ਲੱਦੋ_tensor_q(ggml_backend_zdnn_ਬਫਰ * ਬਫਰ, ਅਚਲ ggml_tensor * tensor);

ਖਾਲੀ ggml_zdnn_init_tensor(ggml_backend_zdnn_ਬਫਰ * ਬਫਰ, ਅਚਲ ggml_tensor * tensor);

#ਜੇ_ਅੰਤ  // GGML_ZDNN_UTILITIES_HPP
