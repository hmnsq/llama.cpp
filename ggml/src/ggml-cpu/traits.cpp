#समावेश "traits.h"

#समावेश "ggml-backend-impl.h"
#समावेश "ggml-backend.h"

namespace ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_बफर_type::~extra_बफर_type() {}
}  // namespace ggml::cpu

bool ggml_cpu_extra_compute_forward(काष्ठा ggml_compute_params * params, काष्ठा ggml_tensor * op) {
    क्रम (स्वतः extra : ggml_backend_cpu_get_extra_बफर_types()) {
        अगर (extra && extra->context) {
            स्वतः buf_extra     = (ggml::cpu::extra_बफर_type *) extra->context;
            स्वतः tensor_traits = buf_extra->get_tensor_traits(op);
            अगर (tensor_traits && tensor_traits->compute_forward(params, op)) {
                वापस true;
            }
        }
    }
    वापस false;
}

bool ggml_cpu_extra_work_आकार(पूर्णांक n_threads, स्थिर काष्ठा ggml_tensor * op, माप_प्रकार * size) {
    क्रम (स्वतः extra : ggml_backend_cpu_get_extra_बफर_types()) {
        अगर (extra && extra->context) {
            स्वतः buf_extra     = (ggml::cpu::extra_बफर_type *) extra->context;
            स्वतः tensor_traits = buf_extra->get_tensor_traits(op);
            अगर (tensor_traits && tensor_traits->work_आकार(n_threads, op, *size)) {
                वापस true;
            }
        }
    }
    वापस false;
}
