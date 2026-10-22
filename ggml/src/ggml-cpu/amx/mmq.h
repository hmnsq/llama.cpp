#ಸೂಚನೆ once
#ಅನ್ತರ್ಭಾವ "common.h"

size_t ggml_backend_amx_desired_wsize(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst);

size_t ggml_backend_amx_get_alloc_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor);

ಶೂನ್ಯ ggml_backend_amx_convert_ತೂಕ(ರಚನೆ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ, size_t ಗಾತ್ರ_2);

ಶೂನ್ಯ ggml_backend_amx_mul_mat(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * dst);
