#ಅನ್ತರ್ಭಾವ "traits.h"

#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"

ನಾಮಾಕಾಶ ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

ತಾರ್ಕಿಕ ggml_cpu_extra_compute_forward(ಸಂರಚನಾ ggml_compute_params * params, ಸಂರಚನಾ ggml_tensor * op) {
    ಪ್ರತಿ (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        ಯದಿ (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            ಯದಿ (tensor_traits && tensor_traits->compute_forward(params, op)) {
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
        }
    }
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

ತಾರ್ಕಿಕ ggml_cpu_extra_work_size(ಪೂರ್ಣಾಙ್ಕ n_threads, ಧ್ರುವ ಸಂರಚನಾ ggml_tensor * op, size_t * size) {
    ಪ್ರತಿ (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        ಯದಿ (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            ಯದಿ (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            }
        }
    }
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}
