#pragma once
#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-cpu-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml.h"

#ifdef __cplusplus
#    include <vector>
ବାହ୍ୟ "C" {
#endif

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * op);
bool ggml_cpu_extra_work_size(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads, ଅଚଳ ସଂରଚନା ggml_tensor * op, size_t * size);

#ifdef __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  ସର୍ବସାଧାରଣ:
    ଆଭାସୀ ~tensor_traits();
    ଆଭାସୀ bool work_size(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads, ଅଚଳ ସଂରଚନା ggml_tensor * op, size_t & size)        = 0;
    ଆଭାସୀ bool compute_forward(ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * op) = 0;
};

class extra_buffer_type {
  ସର୍ବସାଧାରଣ:
    ଆଭାସୀ ~extra_buffer_type();
    ଆଭାସୀ bool            supports_op(ggml_backend_dev_t dev, ଅଚଳ ସଂରଚନା ggml_tensor * op) = 0;
    ଆଭାସୀ tensor_traits * get_tensor_traits(ଅଚଳ ସଂରଚନା ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types();

#endif
