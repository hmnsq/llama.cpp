#pragma once
#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-cpu-impl.h"
#অন্তর্ভাব "ggml.h"

#ifdef __cplusplus
#    include <vector>
বাহ্যিক "C" {
#endif

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * op);
bool ggml_cpu_extra_work_আকার(পূর্ণসংখ্যা n_threads, ধ্রুবক কাঠামো ggml_tensor * op, size_t * আকার_2);

#ifdef __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  সর্বজনীন:
    আভাসী ~tensor_traits();
    আভাসী bool work_আকার(পূর্ণসংখ্যা n_threads, ধ্রুবক কাঠামো ggml_tensor * op, size_t & আকার_2)        = 0;
    আভাসী bool compute_forward(কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * op) = 0;
};

class extra_বাফার_type {
  সর্বজনীন:
    আভাসী ~extra_বাফার_type();
    আভাসী bool            supports_op(ggml_backend_dev_t dev, ধ্রুবক কাঠামো ggml_tensor * op) = 0;
    আভাসী tensor_traits * get_tensor_traits(ধ্রুবক কাঠামো ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::vector<ggml_backend_বাফার_type_t> & ggml_backend_cpu_get_extra_বাফার_types();

#endif
