#અન્તર્ભાવ "traits.h"

#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-backend.h"

નામાકાશ ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

તાર્કિક ggml_cpu_extra_compute_forward(સંરચના ggml_compute_params * params, સંરચના ggml_tensor * op) {
    માટે (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        જો (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            જો (tensor_traits && tensor_traits->compute_forward(params, op)) {
                પાછા સાચું;
            }
        }
    }
    પાછા ખોટું;
}

તાર્કિક ggml_cpu_extra_work_size(પૂર્ણાંક n_threads, અચલ સંરચના ggml_tensor * op, size_t * size) {
    માટે (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        જો (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            જો (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                પાછા સાચું;
            }
        }
    }
    પાછા ખોટું;
}
