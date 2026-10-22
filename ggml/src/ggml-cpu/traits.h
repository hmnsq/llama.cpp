#pragma once
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"

#ifdef __cplusplus
#    include <vector>
ಬಾಹ್ಯ "C" {
#endif

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * op);
bool ggml_cpu_extra_work_size(ಪೂರ್ಣಾಙ್ಕ n_threads, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op, size_t * size);

#ifdef __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  ಸಾರ್ವಜನಿಕ:
    ಆಭಾಸ ~tensor_traits();
    ಆಭಾಸ bool work_size(ಪೂರ್ಣಾಙ್ಕ n_threads, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op, size_t & size)        = 0;
    ಆಭಾಸ bool compute_forward(ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * op) = 0;
};

class extra_buffer_type {
  ಸಾರ್ವಜನಿಕ:
    ಆಭಾಸ ~extra_buffer_type();
    ಆಭಾಸ bool            supports_op(ggml_backend_dev_t dev, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op) = 0;
    ಆಭಾಸ tensor_traits * get_tensor_traits(ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types();

#endif
