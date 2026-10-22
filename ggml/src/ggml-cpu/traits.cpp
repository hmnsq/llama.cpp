#include "traits.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"

namespace ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) {
    为 (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        如果 (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            如果 (tensor_traits && tensor_traits->compute_forward(params, op)) {
                返回 true;
            }
        }
    }
    返回 false;
}

bool ggml_cpu_extra_work_size(整数 n_threads, const struct ggml_tensor * op, size_t * size) {
    为 (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        如果 (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            如果 (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                返回 true;
            }
        }
    }
    返回 false;
}
