#pragma once
#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-cpu-impl.h"
#અન્તર્ભાવ "ggml.h"

#ifdef __cplusplus
#    include <vector>
બાહ્ય "C" {
#endif

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(સંરચના ggml_compute_params * params, સંરચના ggml_tensor * op);
bool ggml_cpu_extra_work_size(પૂર્ણાંક n_threads, અચલ સંરચના ggml_tensor * op, size_t * size);

#ifdef __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  જાહેર:
    આભાસી ~tensor_traits();
    આભાસી bool work_size(પૂર્ણાંક n_threads, અચલ સંરચના ggml_tensor * op, size_t & size)        = 0;
    આભાસી bool compute_forward(સંરચના ggml_compute_params * params, સંરચના ggml_tensor * op) = 0;
};

class extra_buffer_type {
  જાહેર:
    આભાસી ~extra_buffer_type();
    આભાસી bool            supports_op(ggml_backend_dev_t dev, અચલ સંરચના ggml_tensor * op) = 0;
    આભાસી tensor_traits * get_tensor_traits(અચલ સંરચના ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types();

#endif
