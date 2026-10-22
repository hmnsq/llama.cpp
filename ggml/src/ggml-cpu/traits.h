#pragma once
#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-cpu-impl.h"
#അന്തര്ഭാവ "ggml.h"

#ifdef __cplusplus
#    include <vector>
ബാഹ്യ "C" {
#endif

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(ഘടന ggml_compute_params * params, ഘടന ggml_tensor * op);
bool ggml_cpu_extra_work_size(പൂർണ്ണസംഖ്യ n_threads, സ്ഥിരം ഘടന ggml_tensor * op, size_t * size);

#ifdef __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  പൊതു:
    മായാ ~tensor_traits();
    മായാ bool work_size(പൂർണ്ണസംഖ്യ n_threads, സ്ഥിരം ഘടന ggml_tensor * op, size_t & size)        = 0;
    മായാ bool compute_forward(ഘടന ggml_compute_params * params, ഘടന ggml_tensor * op) = 0;
};

class extra_buffer_type {
  പൊതു:
    മായാ ~extra_buffer_type();
    മായാ bool            supports_op(ggml_backend_dev_t dev, സ്ഥിരം ഘടന ggml_tensor * op) = 0;
    മായാ tensor_traits * get_tensor_traits(സ്ഥിരം ഘടന ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types();

#endif
