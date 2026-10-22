#अन्तर्भाव "traits.h"

#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-backend.h"

नामाकाश ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

तार्किक ggml_cpu_extra_compute_forward(संरचना ggml_compute_params * params, संरचना ggml_tensor * op) {
    अत्थाय (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        सचे (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            सचे (tensor_traits && tensor_traits->compute_forward(params, op)) {
                पटिनिवत्त सच्च;
            }
        }
    }
    पटिनिवत्त मिच्छा;
}

तार्किक ggml_cpu_extra_work_size(पूर्णाङ्क n_threads, ध्रुव संरचना ggml_tensor * op, size_t * size) {
    अत्थाय (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        सचे (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            सचे (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                पटिनिवत्त सच्च;
            }
        }
    }
    पटिनिवत्त मिच्छा;
}
