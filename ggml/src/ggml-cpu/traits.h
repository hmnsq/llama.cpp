#pragma once
#অন্তর্ভাৱ "ggml-backend-impl.h"
#অন্তর্ভাৱ "ggml-cpu-impl.h"
#অন্তর্ভাৱ "ggml.h"

#ifdef __cplusplus
#    include <vector>
বাহ্য "C" {
#endif

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * op);
bool ggml_cpu_extra_work_size(পূর্ণাঙ্ক n_threads, ধ্রুৱ সংরচনা ggml_tensor * op, size_t * size);

#ifdef __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  সার্ৱজনিক:
    আভাস ~tensor_traits();
    আভাস bool work_size(পূর্ণাঙ্ক n_threads, ধ্রুৱ সংরচনা ggml_tensor * op, size_t & size)        = 0;
    আভাস bool compute_forward(সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * op) = 0;
};

class extra_buffer_type {
  সার্ৱজনিক:
    আভাস ~extra_buffer_type();
    আভাস bool            supports_op(ggml_backend_dev_t dev, ধ্রুৱ সংরচনা ggml_tensor * op) = 0;
    আভাস tensor_traits * get_tensor_traits(ধ্রুৱ সংরচনা ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types();

#endif
