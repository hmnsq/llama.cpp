#അന്തര്ഭാവ "traits.h"

#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-backend.h"

നാമാകാശ ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

തർക്കം ggml_cpu_extra_compute_forward(ഘടന ggml_compute_params * params, ഘടന ggml_tensor * op) {
    വേണ്ടി (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        എങ്കിൽ (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            എങ്കിൽ (tensor_traits && tensor_traits->compute_forward(params, op)) {
                തിരികെ സത്യം;
            }
        }
    }
    തിരികെ അസത്യം;
}

തർക്കം ggml_cpu_extra_work_size(പൂർണ്ണസംഖ്യ n_threads, സ്ഥിരം ഘടന ggml_tensor * op, size_t * size) {
    വേണ്ടി (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        എങ്കിൽ (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            എങ്കിൽ (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                തിരികെ സത്യം;
            }
        }
    }
    തിരികെ അസത്യം;
}
