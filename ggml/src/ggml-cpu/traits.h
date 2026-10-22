#निर्देशः once
#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-cpu-impl.h"
#अन्तर्भाव "ggml.h"

#यदि_परिभाषितम् __cplusplus
#    include <vector>
बाह्य "C" {
#यदि_अन्तः

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(संरचना ggml_compute_params * params, संरचना ggml_tensor * op);
bool ggml_cpu_extra_work_आकार(पूर्णाङ्क n_तन्तवः, ध्रुव संरचना ggml_tensor * op, size_t * आकार);

#यदि_परिभाषितम् __cplusplus
}

namespace ggml::cpu {
// register in tensor->extra
class tensor_traits {
  सार्वजनिक:
    आभास ~tensor_traits();
    आभास bool work_आकार(पूर्णाङ्क n_तन्तवः, ध्रुव संरचना ggml_tensor * op, size_t & आकार)        = 0;
    आभास bool compute_forward(संरचना ggml_compute_params * params, संरचना ggml_tensor * op) = 0;
};

class extra_बफर_type {
  सार्वजनिक:
    आभास ~extra_बफर_type();
    आभास bool            supports_op(ggml_backend_dev_t dev, ध्रुव संरचना ggml_tensor * op) = 0;
    आभास tensor_traits * get_tensor_traits(ध्रुव संरचना ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::सदिशम्<ggml_backend_बफर_type_t> & ggml_backend_cpu_get_extra_बफर_types();

#यदि_अन्तः
