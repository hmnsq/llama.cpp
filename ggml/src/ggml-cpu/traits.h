#ಸೂಚನೆ once
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
#    include <vector>
ಬಾಹ್ಯ "C" {
#ಆದರೆ_ಅಂತ್ಯ

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * op);
bool ggml_cpu_extra_work_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op, size_t * ಗಾತ್ರ_2);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  ಸಾರ್ವಜನಿಕ:
    ವಾಸ್ತವಿಕ ~tensor_traits();
    ವಾಸ್ತವಿಕ bool work_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op, size_t & ಗಾತ್ರ_2)        = 0;
    ವಾಸ್ತವಿಕ bool compute_forward(ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * op) = 0;
};

class extra_ಬಫರ್_type {
  ಸಾರ್ವಜನಿಕ:
    ವಾಸ್ತವಿಕ ~extra_ಬಫರ್_type();
    ವಾಸ್ತವಿಕ bool            supports_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op) = 0;
    ವಾಸ್ತವಿಕ tensor_traits * get_tensor_traits(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::ಸದಿಶ<ggml_backend_ಬಫರ್_type_t> & ggml_backend_cpu_get_extra_ಬಫರ್_types();

#ಆದರೆ_ಅಂತ್ಯ
