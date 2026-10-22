#अन्तर्भाव "traits.h"

#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-backend.h"

नामाकाश ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

तार्किक ggml_cpu_extra_compute_forward(संरचना ggml_compute_params * params, संरचना ggml_tensor * op) {
    प्रति (स्वतः extra : ggml_backend_cpu_get_extra_buffer_types()) {
        यदि (extra && extra->context) {
            स्वतः buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            स्वतः tensor_traits = buf_extra->get_tensor_traits(op);
            यदि (tensor_traits && tensor_traits->compute_forward(params, op)) {
                प्रत्यागम सत्य;
            }
        }
    }
    प्रत्यागम असत्य;
}

तार्किक ggml_cpu_extra_work_size(पूर्णाङ्क n_threads, ध्रुव संरचना ggml_tensor * op, माप_प्रकार * size) {
    प्रति (स्वतः extra : ggml_backend_cpu_get_extra_buffer_types()) {
        यदि (extra && extra->context) {
            स्वतः buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            स्वतः tensor_traits = buf_extra->get_tensor_traits(op);
            यदि (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                प्रत्यागम सत्य;
            }
        }
    }
    प्रत्यागम असत्य;
}
