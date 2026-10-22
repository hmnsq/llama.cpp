#ifndef GGML_ZDNN_UTILITIES_HPP
#define GGML_ZDNN_UTILITIES_HPP

#அந்தர்பாவ "common.hpp"

zdnn_data_types ggml_zdnn_type_mapping(ggml_type type);

வெறுமை ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      மாறிலி ggml_tensor       * src,
                      மாறிலி int64_t           * ne,
                      மாறிலி zdnn_data_layouts   layout);

வெறுமை ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, வெறுமை * buffer);

int64_t ggml_zdnn_q_group_size(மாறிலி ggml_tensor * tensor);

வெறுமை ggml_zdnn_load_tensor_q(ggml_backend_zdnn_buffer * buffer, மாறிலி ggml_tensor * tensor);

வெறுமை ggml_zdnn_init_tensor(ggml_backend_zdnn_buffer * buffer, மாறிலி ggml_tensor * tensor);

#endif  // GGML_ZDNN_UTILITIES_HPP
