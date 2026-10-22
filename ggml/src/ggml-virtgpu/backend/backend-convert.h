#ضمن "shared/apir_backend.h"

#تعریف_کړه بفر_منزل_HOST_HANDLE(name) ggml_بفر_منزل_apir_handle(name)

ساكن مضمن apir_بفر_host_handle_t ggml_بفر_منزل_apir_handle(ggml_backend_بفر_t بفر) {
    // in the backend, the buffer handle is the buffer pointer
    أرجع (apir_بفر_host_handle_t) بفر;
}

ساكن مضمن apir_بفر_type_host_handle_t ggml_بفر_type_منزل_apir_handle(ggml_backend_بفر_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    أرجع (apir_بفر_type_host_handle_t) buft;
}
