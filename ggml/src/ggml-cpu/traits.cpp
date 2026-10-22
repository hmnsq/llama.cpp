#ಅನ್ತರ್ಭಾವ "traits.h"

#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"

ನಾಮಾಕಾಶ ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_ಬಫರ್_type::~extra_ಬಫರ್_type() {}
}  // namespace ggml::cpu

ತಾರ್ಕಿಕ ggml_cpu_extra_compute_forward(ರಚನೆ ggml_compute_params * params, ರಚನೆ ggml_tensor * op) {
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ extra : ggml_backend_cpu_get_extra_ಬಫರ್_types()) {
        ಆದರೆ (extra && extra->context) {
            ಸ್ವಯಂಚಾಲಿತ buf_extra     = (ggml::cpu::extra_ಬಫರ್_type *) extra->context;
            ಸ್ವಯಂಚಾಲಿತ tensor_traits = buf_extra->get_tensor_traits(op);
            ಆದರೆ (tensor_traits && tensor_traits->compute_forward(params, op)) {
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        }
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ತಾರ್ಕಿಕ ggml_cpu_extra_work_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು, ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * op, size_t * size) {
    ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ extra : ggml_backend_cpu_get_extra_ಬಫರ್_types()) {
        ಆದರೆ (extra && extra->context) {
            ಸ್ವಯಂಚಾಲಿತ buf_extra     = (ggml::cpu::extra_ಬಫರ್_type *) extra->context;
            ಸ್ವಯಂಚಾಲಿತ tensor_traits = buf_extra->get_tensor_traits(op);
            ಆದರೆ (tensor_traits && tensor_traits->work_ಗಾತ್ರ(n_ಎಳೆಗಳು, op, *size)) {
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        }
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}
