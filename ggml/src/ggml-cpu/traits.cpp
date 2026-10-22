#অন্তর্ভাব "traits.h"

#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-backend.h"

নামাকাশ ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_বাফার_type::~extra_বাফার_type() {}
}  // namespace ggml::cpu

যৌক্তিক ggml_cpu_extra_compute_forward(কাঠামো ggml_compute_params * params, কাঠামো ggml_tensor * op) {
    জন্য (auto extra : ggml_backend_cpu_get_extra_বাফার_types()) {
        যদি (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_বাফার_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            যদি (tensor_traits && tensor_traits->compute_forward(params, op)) {
                ফেরত সত্য;
            }
        }
    }
    ফেরত মিথ্যা;
}

যৌক্তিক ggml_cpu_extra_work_আকার(পূর্ণসংখ্যা n_threads, ধ্রুবক কাঠামো ggml_tensor * op, size_t * size) {
    জন্য (auto extra : ggml_backend_cpu_get_extra_বাফার_types()) {
        যদি (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_বাফার_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            যদি (tensor_traits && tensor_traits->work_আকার(n_threads, op, *size)) {
                ফেরত সত্য;
            }
        }
    }
    ফেরত মিথ্যা;
}
