#include "traits.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"

namespace ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

bool ggml_cpu_extra_compute_forward(sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * op) {
    සඳහා (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        නම් (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            නම් (tensor_traits && tensor_traits->compute_forward(params, op)) {
                ආපසු true;
            }
        }
    }
    ආපසු false;
}

bool ggml_cpu_extra_work_size(පූර්ණ_සංඛ්‍යා n_threads, const sa_nracanaa ggml_tensor * op, size_t * size) {
    සඳහා (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        නම් (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            නම් (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                ආපසු true;
            }
        }
    }
    ආපසු false;
}
