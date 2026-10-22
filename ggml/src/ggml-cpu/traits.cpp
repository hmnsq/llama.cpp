#include "traits.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"

namespace ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) {
    ために (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        もし (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            もし (tensor_traits && tensor_traits->compute_forward(params, op)) {
                戻る true;
            }
        }
    }
    戻る false;
}

bool ggml_cpu_extra_work_size(整数 n_threads, const struct ggml_tensor * op, size_t * size) {
    ために (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        もし (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            もし (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                戻る true;
            }
        }
    }
    戻る false;
}
