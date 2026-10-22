#आशय once
#समावेश "ggml-backend-impl.h"
#समावेश "ggml-cpu-impl.h"
#समावेश "ggml.h"

#अगर_घोषित __cplusplus
#    include <vector>
बाह्य "C" {
#पूर्ण_अगर

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * op);
bool ggml_cpu_extra_work_आकार(पूर्णांक n_threads, स्थिर काष्ठा ggml_tensor * op, माप_प्रकार * आकार);

#अगर_घोषित __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  खुला:
    भव ~tensor_traits();
    भव bool work_आकार(पूर्णांक n_threads, स्थिर काष्ठा ggml_tensor * op, माप_प्रकार & आकार)        = 0;
    भव bool compute_forward(काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * op) = 0;
};

class extra_बफर_type {
  खुला:
    भव ~extra_बफर_type();
    भव bool            supports_op(ggml_backend_dev_t dev, स्थिर काष्ठा ggml_tensor * op) = 0;
    भव tensor_traits * get_tensor_traits(स्थिर काष्ठा ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::vector<ggml_backend_बफर_type_t> & ggml_backend_cpu_get_extra_बफर_types();

#पूर्ण_अगर
