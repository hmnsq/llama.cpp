#অন্তর্ভাৱ "traits.h"

#অন্তর্ভাৱ "ggml-backend-impl.h"
#অন্তর্ভাৱ "ggml-backend.h"

নামাকাশ ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

তার্কিক ggml_cpu_extra_compute_forward(সংরচনা ggml_compute_params * params, সংরচনা ggml_tensor * op) {
    বাবে (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        যদি (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            যদি (tensor_traits && tensor_traits->compute_forward(params, op)) {
                ঘূৰাই সঁচা;
            }
        }
    }
    ঘূৰাই মিছা;
}

তার্কিক ggml_cpu_extra_work_size(পূর্ণাঙ্ক n_threads, ধ্রুৱ সংরচনা ggml_tensor * op, size_t * size) {
    বাবে (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        যদি (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            যদি (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                ঘূৰাই সঁচা;
            }
        }
    }
    ঘূৰাই মিছা;
}
