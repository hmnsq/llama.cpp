#অন্তর্ভাব "shared/apir_backend.h"

#define বাফার_TO_HOST_HANDLE(name) ggml_বাফার_to_apir_handle(name)

স্থির অন্তঃরেখ apir_বাফার_host_handle_t ggml_বাফার_to_apir_handle(ggml_backend_বাফার_t বাফার) {
    // in the backend, the buffer handle is the buffer pointer
    ফেরত (apir_বাফার_host_handle_t) বাফার;
}

স্থির অন্তঃরেখ apir_বাফার_type_host_handle_t ggml_বাফার_type_to_apir_handle(ggml_backend_বাফার_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    ফেরত (apir_বাফার_type_host_handle_t) buft;
}
