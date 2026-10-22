#ਹਦਾਇਤ once
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
#    include <vector>
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * op);
bool ggml_cpu_extra_work_ਮਾਪ(ਪੂਰਨ n_ਧਾਗੇ, ਅਚਲ ਬਣਤਰ ggml_tensor * op, size_t * ਮਾਪ);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  ਜਨਤਕ:
    ਆਭਾਸੀ ~tensor_traits();
    ਆਭਾਸੀ bool work_ਮਾਪ(ਪੂਰਨ n_ਧਾਗੇ, ਅਚਲ ਬਣਤਰ ggml_tensor * op, size_t & ਮਾਪ)        = 0;
    ਆਭਾਸੀ bool compute_forward(ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * op) = 0;
};

class extra_ਬਫਰ_type {
  ਜਨਤਕ:
    ਆਭਾਸੀ ~extra_ਬਫਰ_type();
    ਆਭਾਸੀ bool            supports_op(ggml_backend_dev_t dev, ਅਚਲ ਬਣਤਰ ggml_tensor * op) = 0;
    ਆਭਾਸੀ tensor_traits * get_tensor_traits(ਅਚਲ ਬਣਤਰ ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::ਸਦਿਸ਼<ggml_backend_ਬਫਰ_type_t> & ggml_backend_cpu_get_extra_ਬਫਰ_types();

#ਜੇ_ਅੰਤ
