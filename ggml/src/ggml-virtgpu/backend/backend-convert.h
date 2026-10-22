#شامل "shared/apir_backend.h"

#تعریف_کرو بفر_مقصد_HOST_HANDLE(name) ggml_بفر_مقصد_apir_handle(name)

ساکن اندرونی apir_بفر_host_handle_t ggml_بفر_مقصد_apir_handle(ggml_backend_بفر_t بفر) {
    // in the backend, the buffer handle is the buffer pointer
    واپس (apir_بفر_host_handle_t) بفر;
}

ساکن اندرونی apir_بفر_type_host_handle_t ggml_بفر_type_مقصد_apir_handle(ggml_backend_بفر_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    واپس (apir_بفر_type_host_handle_t) buft;
}
