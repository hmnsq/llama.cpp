#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_ZDNN_UTILITIES_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_ZDNN_UTILITIES_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"

zdnn_ದತ್ತಾಂಶ_types ggml_zdnn_type_mapping(ggml_type type);

ಶೂನ್ಯ ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      ಸ್ಥಿರಾಂಕ ggml_tensor       * src,
                      ಸ್ಥಿರಾಂಕ int64_t           * ne,
                      ಸ್ಥಿರಾಂಕ zdnn_ದತ್ತಾಂಶ_layouts   layout);

ಶೂನ್ಯ ggml_zdnn_ತುಂಬು_tensor(zdnn_ztensor & ztensor, ಶೂನ್ಯ * ಬಫರ್);

int64_t ggml_zdnn_q_ಗುಂಪು_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ ggml_tensor * tensor);

ಶೂನ್ಯ ggml_zdnn_ತುಂಬು_tensor_q(ggml_backend_zdnn_ಬಫರ್ * ಬಫರ್, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor);

ಶೂನ್ಯ ggml_zdnn_init_tensor(ggml_backend_zdnn_ಬಫರ್ * ಬಫರ್, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor);

#ಆದರೆ_ಅಂತ್ಯ  // GGML_ZDNN_UTILITIES_HPP
