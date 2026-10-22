#ضمن "traits.h"

#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"

نطاق ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

منطقي ggml_cpu_extra_compute_forward(بنية ggml_compute_params * params, بنية ggml_tensor * op) {
    لكل (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        إذا (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            إذا (tensor_traits && tensor_traits->compute_forward(params, op)) {
                أرجع true;
            }
        }
    }
    أرجع خطأ;
}

منطقي ggml_cpu_extra_work_size(صحيح n_threads, ثابت بنية ggml_tensor * op, size_t * size) {
    لكل (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        إذا (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            إذا (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                أرجع true;
            }
        }
    }
    أرجع خطأ;
}
