#ਅਨ੍ਤਰ੍ਭਾਵ "shared/apir_backend.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਬਫਰ_ਮੰਜ਼ਿਲ_HOST_HANDLE(name) ggml_ਬਫਰ_ਮੰਜ਼ਿਲ_apir_handle(name)

ਸਥਿਰ ਸਤਰਅੰਦਰ apir_ਬਫਰ_host_handle_t ggml_ਬਫਰ_ਮੰਜ਼ਿਲ_apir_handle(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    // in the backend, the buffer handle is the buffer pointer
    ਵਾਪਸ (apir_ਬਫਰ_host_handle_t) ਬਫਰ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ apir_ਬਫਰ_type_host_handle_t ggml_ਬਫਰ_type_ਮੰਜ਼ਿਲ_apir_handle(ggml_backend_ਬਫਰ_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    ਵਾਪਸ (apir_ਬਫਰ_type_host_handle_t) buft;
}
