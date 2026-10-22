#ضمن "traits.h"

#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"

نطاق ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_میانگیر_type::~extra_میانگیر_type() {}
}  // namespace ggml::cpu

منطقي ggml_cpu_extra_compute_forward(بنية ggml_compute_params * params, بنية ggml_tensor * op) {
    لكل (خودکار extra : ggml_backend_cpu_get_extra_میانگیر_types()) {
        إذا (extra && extra->context) {
            خودکار buf_extra     = (ggml::cpu::extra_میانگیر_type *) extra->context;
            خودکار tensor_traits = buf_extra->get_tensor_traits(op);
            إذا (tensor_traits && tensor_traits->compute_forward(params, op)) {
                أرجع درست;
            }
        }
    }
    أرجع خطأ;
}

منطقي ggml_cpu_extra_work_اندازه(صحيح n_نخها, ثابت بنية ggml_tensor * op, size_t * size) {
    لكل (خودکار extra : ggml_backend_cpu_get_extra_میانگیر_types()) {
        إذا (extra && extra->context) {
            خودکار buf_extra     = (ggml::cpu::extra_میانگیر_type *) extra->context;
            خودکار tensor_traits = buf_extra->get_tensor_traits(op);
            إذا (tensor_traits && tensor_traits->work_اندازه(n_نخها, op, *size)) {
                أرجع درست;
            }
        }
    }
    أرجع خطأ;
}
