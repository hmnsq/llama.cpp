#رهنمود once
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-cpu-impl.h"
#ضمن "ggml.h"

#اگر_تعریف_شده __cplusplus
#    include <vector>
خارجي "C" {
#پایان_اگر

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(بنية ggml_compute_params * params, بنية ggml_tensor * op);
bool ggml_cpu_extra_work_اندازه(صحيح n_نخها, ثابت بنية ggml_tensor * op, size_t * اندازه);

#اگر_تعریف_شده __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  عام:
    مجازی ~tensor_traits();
    مجازی bool work_اندازه(صحيح n_نخها, ثابت بنية ggml_tensor * op, size_t & اندازه)        = 0;
    مجازی bool compute_forward(بنية ggml_compute_params * params, بنية ggml_tensor * op) = 0;
};

class extra_میانگیر_type {
  عام:
    مجازی ~extra_میانگیر_type();
    مجازی bool            supports_op(ggml_backend_dev_t dev, ثابت بنية ggml_tensor * op) = 0;
    مجازی tensor_traits * get_tensor_traits(ثابت بنية ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::وکتور<ggml_backend_میانگیر_type_t> & ggml_backend_cpu_get_extra_میانگیر_types();

#پایان_اگر
