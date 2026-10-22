#pragma once
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-cpu-impl.h"
#ضمن "ggml.h"

#ifdef __cplusplus
#    include <vector>
خارجي "C" {
#endif

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(بنية ggml_compute_params * params, بنية ggml_tensor * op);
bool ggml_cpu_extra_work_size(صحيح n_threads, ثابت بنية ggml_tensor * op, size_t * size);

#ifdef __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  عام:
    virtual ~tensor_traits();
    virtual bool work_size(صحيح n_threads, ثابت بنية ggml_tensor * op, size_t & size)        = 0;
    virtual bool compute_forward(بنية ggml_compute_params * params, بنية ggml_tensor * op) = 0;
};

class extra_buffer_type {
  عام:
    virtual ~extra_buffer_type();
    virtual bool            supports_op(ggml_backend_dev_t dev, ثابت بنية ggml_tensor * op) = 0;
    virtual tensor_traits * get_tensor_traits(ثابت بنية ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types();

#endif
