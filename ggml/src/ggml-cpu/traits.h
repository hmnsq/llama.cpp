#ہدایت once
#شامل "ggml-backend-impl.h"
#شامل "ggml-cpu-impl.h"
#شامل "ggml.h"

#اگر_تعریف_شدہ __cplusplus
#    include <vector>
خارجی "C" {
#اگر_ختم

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(ساخت ggml_compute_params * params, ساخت ggml_tensor * op);
bool ggml_cpu_extra_work_سائز(صحیح n_دھاگے, ثابت ساخت ggml_tensor * op, size_t * سائز);

#اگر_تعریف_شدہ __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  عوامی:
    مجازی ~tensor_traits();
    مجازی bool work_سائز(صحیح n_دھاگے, ثابت ساخت ggml_tensor * op, size_t & سائز)        = 0;
    مجازی bool compute_forward(ساخت ggml_compute_params * params, ساخت ggml_tensor * op) = 0;
};

class extra_بفر_type {
  عوامی:
    مجازی ~extra_بفر_type();
    مجازی bool            supports_op(ggml_backend_dev_t dev, ثابت ساخت ggml_tensor * op) = 0;
    مجازی tensor_traits * get_tensor_traits(ثابت ساخت ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::سمتیہ<ggml_backend_بفر_type_t> & ggml_backend_cpu_get_extra_بفر_types();

#اگر_ختم
