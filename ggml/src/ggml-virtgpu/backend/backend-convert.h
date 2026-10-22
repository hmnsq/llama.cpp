#ضمن "shared/apir_backend.h"

#تعریف_کن میانگیر_مقصد_HOST_HANDLE(name) ggml_میانگیر_مقصد_apir_handle(name)

ساكن مضمن apir_میانگیر_host_handle_t ggml_میانگیر_مقصد_apir_handle(ggml_backend_میانگیر_t میانگیر) {
    // in the backend, the buffer handle is the buffer pointer
    أرجع (apir_میانگیر_host_handle_t) میانگیر;
}

ساكن مضمن apir_میانگیر_type_host_handle_t ggml_میانگیر_type_مقصد_apir_handle(ggml_backend_میانگیر_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    أرجع (apir_میانگیر_type_host_handle_t) buft;
}
