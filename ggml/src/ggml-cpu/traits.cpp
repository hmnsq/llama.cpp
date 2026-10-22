#شامل "traits.h"

#شامل "ggml-backend-impl.h"
#شامل "ggml-backend.h"

نامخلا ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_بفر_type::~extra_بفر_type() {}
}  // namespace ggml::cpu

منطقی ggml_cpu_extra_compute_forward(ساخت ggml_compute_params * params, ساخت ggml_tensor * op) {
    برائے (خودکار extra : ggml_backend_cpu_get_extra_بفر_types()) {
        اگر (extra && extra->context) {
            خودکار buf_extra     = (ggml::cpu::extra_بفر_type *) extra->context;
            خودکار tensor_traits = buf_extra->get_tensor_traits(op);
            اگر (tensor_traits && tensor_traits->compute_forward(params, op)) {
                واپس سچ;
            }
        }
    }
    واپس جھوٹ;
}

منطقی ggml_cpu_extra_work_سائز(صحیح n_دھاگے, ثابت ساخت ggml_tensor * op, size_t * size) {
    برائے (خودکار extra : ggml_backend_cpu_get_extra_بفر_types()) {
        اگر (extra && extra->context) {
            خودکار buf_extra     = (ggml::cpu::extra_بفر_type *) extra->context;
            خودکار tensor_traits = buf_extra->get_tensor_traits(op);
            اگر (tensor_traits && tensor_traits->work_سائز(n_دھاگے, op, *size)) {
                واپس سچ;
            }
        }
    }
    واپس جھوٹ;
}
