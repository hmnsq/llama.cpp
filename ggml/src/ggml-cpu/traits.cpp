#అన్తర్భావ "traits.h"

#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-backend.h"

నామాకాశ ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_బఫరు_type::~extra_బఫరు_type() {}
}  // namespace ggml::cpu

తార్కిక ggml_cpu_extra_compute_forward(నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * op) {
    కొరకు (స్వయంచాలక extra : ggml_backend_cpu_get_extra_బఫరు_types()) {
        ఒకవేళ (extra && extra->context) {
            స్వయంచాలక buf_extra     = (ggml::cpu::extra_బఫరు_type *) extra->context;
            స్వయంచాలక tensor_traits = buf_extra->get_tensor_traits(op);
            ఒకవేళ (tensor_traits && tensor_traits->compute_forward(params, op)) {
                తిరిగి నిజం;
            }
        }
    }
    తిరిగి అబద్ధం;
}

తార్కిక ggml_cpu_extra_work_కొలత(పూర్ణాంకం n_దారాలు, స్థిరరాశి నిర్మాణం ggml_tensor * op, size_t * size) {
    కొరకు (స్వయంచాలక extra : ggml_backend_cpu_get_extra_బఫరు_types()) {
        ఒకవేళ (extra && extra->context) {
            స్వయంచాలక buf_extra     = (ggml::cpu::extra_బఫరు_type *) extra->context;
            స్వయంచాలక tensor_traits = buf_extra->get_tensor_traits(op);
            ఒకవేళ (tensor_traits && tensor_traits->work_కొలత(n_దారాలు, op, *size)) {
                తిరిగి నిజం;
            }
        }
    }
    తిరిగి అబద్ధం;
}
