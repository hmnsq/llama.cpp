#شامل "traits.h"

#شامل "ggml-backend-impl.h"
#شامل "ggml-backend.h"

فضاینام ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

منطقی ggml_cpu_extra_compute_forward(ساختار ggml_compute_params * params, ساختار ggml_tensor * op) {
    برای (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        اگر (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            اگر (tensor_traits && tensor_traits->compute_forward(params, op)) {
                بازگشت درست;
            }
        }
    }
    بازگشت نادرست;
}

منطقی ggml_cpu_extra_work_size(صحیح n_threads, ثابت ساختار ggml_tensor * op, size_t * size) {
    برای (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        اگر (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            اگر (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                بازگشت درست;
            }
        }
    }
    بازگشت نادرست;
}
