#ଅନ୍ତର୍ଭାଵ "traits.h"

#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend.h"

ନାମାକାଶ ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

ତାର୍କିକ ggml_cpu_extra_compute_forward(ସଂରଚନା ggml_compute_params * params, ସଂରଚନା ggml_tensor * op) {
    ପାଇଁ (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        ଯଦି (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            ଯଦି (tensor_traits && tensor_traits->compute_forward(params, op)) {
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
        }
    }
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

ତାର୍କିକ ggml_cpu_extra_work_size(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads, ଅଚଳ ସଂରଚନା ggml_tensor * op, size_t * size) {
    ପାଇଁ (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        ଯଦି (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            ଯଦି (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
        }
    }
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}
