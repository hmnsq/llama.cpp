#pragma once
#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-cpu-impl.h"
#அந்தர்பாவ "ggml.h"

#ifdef __cplusplus
#    include <vector>
வெளி "C" {
#endif

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * op);
bool ggml_cpu_extra_work_size(முழு n_threads, மாறிலி கட்டமைப்பு ggml_tensor * op, size_t * size);

#ifdef __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  பொது:
    மெய்நிகர் ~tensor_traits();
    மெய்நிகர் bool work_size(முழு n_threads, மாறிலி கட்டமைப்பு ggml_tensor * op, size_t & size)        = 0;
    மெய்நிகர் bool compute_forward(கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * op) = 0;
};

class extra_buffer_type {
  பொது:
    மெய்நிகர் ~extra_buffer_type();
    மெய்நிகர் bool            supports_op(ggml_backend_dev_t dev, மாறிலி கட்டமைப்பு ggml_tensor * op) = 0;
    மெய்நிகர் tensor_traits * get_tensor_traits(மாறிலி கட்டமைப்பு ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types();

#endif
