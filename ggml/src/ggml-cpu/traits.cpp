#ਅਨ੍ਤਰ੍ਭਾਵ "traits.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"

ਨਾਮਾਕਾਸ਼ ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_ਬਫਰ_type::~extra_ਬਫਰ_type() {}
}  // namespace ggml::cpu

ਤਾਰਕਿਕ ggml_cpu_extra_compute_forward(ਬਣਤਰ ggml_compute_params * params, ਬਣਤਰ ggml_tensor * op) {
    ਲਈ (ਸਵੈਚਾਲਿਤ extra : ggml_backend_cpu_get_extra_ਬਫਰ_types()) {
        ਜੇ (extra && extra->context) {
            ਸਵੈਚਾਲਿਤ buf_extra     = (ggml::cpu::extra_ਬਫਰ_type *) extra->context;
            ਸਵੈਚਾਲਿਤ tensor_traits = buf_extra->get_tensor_traits(op);
            ਜੇ (tensor_traits && tensor_traits->compute_forward(params, op)) {
                ਵਾਪਸ ਸੱਚ;
            }
        }
    }
    ਵਾਪਸ ਝੂਠ;
}

ਤਾਰਕਿਕ ggml_cpu_extra_work_ਮਾਪ(ਪੂਰਨ n_ਧਾਗੇ, ਅਚਲ ਬਣਤਰ ggml_tensor * op, size_t * size) {
    ਲਈ (ਸਵੈਚਾਲਿਤ extra : ggml_backend_cpu_get_extra_ਬਫਰ_types()) {
        ਜੇ (extra && extra->context) {
            ਸਵੈਚਾਲਿਤ buf_extra     = (ggml::cpu::extra_ਬਫਰ_type *) extra->context;
            ਸਵੈਚਾਲਿਤ tensor_traits = buf_extra->get_tensor_traits(op);
            ਜੇ (tensor_traits && tensor_traits->work_ਮਾਪ(n_ਧਾਗੇ, op, *size)) {
                ਵਾਪਸ ਸੱਚ;
            }
        }
    }
    ਵਾਪਸ ਝੂਠ;
}
