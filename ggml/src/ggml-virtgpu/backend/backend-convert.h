#அந்தர்பாவ "shared/apir_backend.h"

#define BUFFER_TO_HOST_HANDLE(name) ggml_buffer_to_apir_handle(name)

நிலையான உள்வரி apir_buffer_host_handle_t ggml_buffer_to_apir_handle(ggml_backend_buffer_t buffer) {
    // in the backend, the buffer handle is the buffer pointer
    திரும்பு (apir_buffer_host_handle_t) buffer;
}

நிலையான உள்வரி apir_buffer_type_host_handle_t ggml_buffer_type_to_apir_handle(ggml_backend_buffer_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    திரும்பு (apir_buffer_type_host_handle_t) buft;
}
