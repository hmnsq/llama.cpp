#pragma once
#شامل "ggml-backend-impl.h"
#شامل "ggml-cpu-impl.h"
#شامل "ggml.h"

#ifdef __cplusplus
#    include <vector>
بیرونی "C" {
#endif

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(ساختار ggml_compute_params * params, ساختار ggml_tensor * op);
bool ggml_cpu_extra_work_size(صحیح n_threads, ثابت ساختار ggml_tensor * op, size_t * size);

#ifdef __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  عمومی:
    مجازی ~tensor_traits();
    مجازی bool work_size(صحیح n_threads, ثابت ساختار ggml_tensor * op, size_t & size)        = 0;
    مجازی bool compute_forward(ساختار ggml_compute_params * params, ساختار ggml_tensor * op) = 0;
};

class extra_buffer_type {
  عمومی:
    مجازی ~extra_buffer_type();
    مجازی bool            supports_op(ggml_backend_dev_t dev, ثابت ساختار ggml_tensor * op) = 0;
    مجازی tensor_traits * get_tensor_traits(ثابت ساختار ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types();

#endif
