#هدايت once
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-cpu-impl.h"
#ضمن "ggml.h"

#جيڪڏهن_وصف_ٿيل __cplusplus
#    include <vector>
خارجي "C" {
#جيڪڏهن_پورو

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(بنية ggml_compute_params * params, بنية ggml_tensor * op);
bool ggml_cpu_extra_work_سائيز(صحيح n_ڌاڳا, ثابت بنية ggml_tensor * op, size_t * سائيز);

#جيڪڏهن_وصف_ٿيل __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  عام:
    مجازي ~tensor_traits();
    مجازي bool work_سائيز(صحيح n_ڌاڳا, ثابت بنية ggml_tensor * op, size_t & سائيز)        = 0;
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
std::ويڪٽر<ggml_backend_بفر_type_t> & ggml_backend_cpu_get_extra_بفر_types();

#جيڪڏهن_پورو
