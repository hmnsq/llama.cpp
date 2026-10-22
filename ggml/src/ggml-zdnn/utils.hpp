#ifndef GGML_ZDNN_UTILITIES_HPP
#define GGML_ZDNN_UTILITIES_HPP

#অন্তর্ভাব "common.hpp"

zdnn_তথ্য_types ggml_zdnn_type_mapping(ggml_type type);

শূন্য ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      ধ্রুবক ggml_tensor       * src,
                      ধ্রুবক int64_t           * ne,
                      ধ্রুবক zdnn_তথ্য_layouts   layout);

শূন্য ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, শূন্য * বাফার);

int64_t ggml_zdnn_q_group_আকার(ধ্রুবক ggml_tensor * tensor);

শূন্য ggml_zdnn_load_tensor_q(ggml_backend_zdnn_বাফার * বাফার, ধ্রুবক ggml_tensor * tensor);

শূন্য ggml_zdnn_init_tensor(ggml_backend_zdnn_বাফার * বাফার, ধ্রুবক ggml_tensor * tensor);

#endif  // GGML_ZDNN_UTILITIES_HPP
