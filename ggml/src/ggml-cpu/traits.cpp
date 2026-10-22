#include "traits.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"

namespace ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

bool ggml_cpu_extra_compute_forward(sa_nracanaa ggml_compute_params * params, sa_nracanaa ggml_tensor * op) {
    དོན་ལུ (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        གལ་སྲིད (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            གལ་སྲིད (tensor_traits && tensor_traits->compute_forward(params, op)) {
                ལོག true;
            }
        }
    }
    ལོག false;
}

bool ggml_cpu_extra_work_size(ཧྲིལ n_threads, const sa_nracanaa ggml_tensor * op, size_t * size) {
    དོན་ལུ (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        གལ་སྲིད (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            གལ་སྲིད (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                ལོག true;
            }
        }
    }
    ལོག false;
}
