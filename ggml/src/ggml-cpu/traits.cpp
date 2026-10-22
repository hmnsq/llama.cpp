#அந்தர்பாவ "traits.h"

#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-backend.h"

நாமாகாஶ ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

தருக்க ggml_cpu_extra_compute_forward(கட்டமைப்பு ggml_compute_params * params, கட்டமைப்பு ggml_tensor * op) {
    க்கு (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        எனில் (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            எனில் (tensor_traits && tensor_traits->compute_forward(params, op)) {
                திரும்பு உண்மை;
            }
        }
    }
    திரும்பு பொய்;
}

தருக்க ggml_cpu_extra_work_size(முழு n_threads, மாறிலி கட்டமைப்பு ggml_tensor * op, size_t * size) {
    க்கு (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        எனில் (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            எனில் (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                திரும்பு உண்மை;
            }
        }
    }
    திரும்பு பொய்;
}
