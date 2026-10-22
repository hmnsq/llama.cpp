#అన్తర్భావ "shared/apir_backend.h"

#నిర్వచించు బఫరు_గమ్యం_HOST_HANDLE(name) ggml_బఫరు_గమ్యం_apir_handle(name)

స్థిర అంతఃవరుస apir_బఫరు_host_handle_t ggml_బఫరు_గమ్యం_apir_handle(ggml_backend_బఫరు_t బఫరు) {
    // in the backend, the buffer handle is the buffer pointer
    తిరిగి (apir_బఫరు_host_handle_t) బఫరు;
}

స్థిర అంతఃవరుస apir_బఫరు_type_host_handle_t ggml_బఫరు_type_గమ్యం_apir_handle(ggml_backend_బఫరు_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    తిరిగి (apir_బఫరు_type_host_handle_t) buft;
}
