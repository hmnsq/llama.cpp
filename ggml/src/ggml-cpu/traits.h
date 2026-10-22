#لارښوونه once
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-cpu-impl.h"
#ضمن "ggml.h"

#که_تعریف_شوی __cplusplus
#    include <vector>
خارجي "C" {
#د_که_پای

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(بنية ggml_compute_params * params, بنية ggml_tensor * op);
bool ggml_cpu_extra_work_اندازه(صحيح n_تارونه, ثابت بنية ggml_tensor * op, size_t * اندازه);

#که_تعریف_شوی __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  عام:
    مجازي ~tensor_traits();
    مجازي bool work_اندازه(صحيح n_تارونه, ثابت بنية ggml_tensor * op, size_t & اندازه)        = 0;
    مجازي bool compute_forward(بنية ggml_compute_params * params, بنية ggml_tensor * op) = 0;
};

class extra_بفر_type {
  عام:
    مجازي ~extra_بفر_type();
    مجازي bool            supports_op(ggml_backend_dev_t dev, ثابت بنية ggml_tensor * op) = 0;
    مجازي tensor_traits * get_tensor_traits(ثابت بنية ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::ویکټور<ggml_backend_بفر_type_t> & ggml_backend_cpu_get_extra_بفر_types();

#د_که_پای
