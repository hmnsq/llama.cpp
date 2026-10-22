#సూచన once
#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-cpu-impl.h"
#అన్తర్భావ "ggml.h"

#నిర్వచితమైతే __cplusplus
#    include <vector>
బాహ్య "C" {
#అయితే_ముగింపు

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * op);
bool ggml_cpu_extra_work_కొలత(పూర్ణాంకం n_దారాలు, స్థిరరాశి నిర్మాణం ggml_tensor * op, size_t * కొలత);

#నిర్వచితమైతే __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  బహిరంగ:
    మాయా ~tensor_traits();
    మాయా bool work_కొలత(పూర్ణాంకం n_దారాలు, స్థిరరాశి నిర్మాణం ggml_tensor * op, size_t & కొలత)        = 0;
    మాయా bool compute_forward(నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * op) = 0;
};

class extra_బఫరు_type {
  బహిరంగ:
    మాయా ~extra_బఫరు_type();
    మాయా bool            supports_op(ggml_backend_dev_t dev, స్థిరరాశి నిర్మాణం ggml_tensor * op) = 0;
    మాయా tensor_traits * get_tensor_traits(స్థిరరాశి నిర్మాణం ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::సదిశ<ggml_backend_బఫరు_type_t> & ggml_backend_cpu_get_extra_బఫరు_types();

#అయితే_ముగింపు
