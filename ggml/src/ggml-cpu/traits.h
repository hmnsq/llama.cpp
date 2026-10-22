#ہدایت once
#ضمن "ggml-backend-impl.h"
#ضمن "ggml-cpu-impl.h"
#ضمن "ggml.h"

#جے_تعریف_شدہ __cplusplus
#    include <vector>
خارجي "C" {
#جے_انت

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(بنية ggml_compute_params * params, بنية ggml_tensor * op);
bool ggml_cpu_extra_work_ناپ(صحيح n_دھاگے, ثابت بنية ggml_tensor * op, size_t * ناپ);

#جے_تعریف_شدہ __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  عام:
    مجازی ~tensor_traits();
    مجازی bool work_ناپ(صحيح n_دھاگے, ثابت بنية ggml_tensor * op, size_t & ناپ)        = 0;
    مجازی bool compute_forward(بنية ggml_compute_params * params, بنية ggml_tensor * op) = 0;
};

class extra_بفر_type {
  عام:
    مجازی ~extra_بفر_type();
    مجازی bool            supports_op(ggml_backend_dev_t dev, ثابت بنية ggml_tensor * op) = 0;
    مجازی tensor_traits * get_tensor_traits(ثابت بنية ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::سمتیہ<ggml_backend_بفر_type_t> & ggml_backend_cpu_get_extra_بفر_types();

#جے_انت
