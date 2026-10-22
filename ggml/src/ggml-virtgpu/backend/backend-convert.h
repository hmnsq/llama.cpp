#ಅನ್ತರ್ಭಾವ "shared/apir_backend.h"

#ವ್ಯಾಖ್ಯಾನಿಸು ಬಫರ್_ಗಮ್ಯ_HOST_HANDLE(name) ggml_ಬಫರ್_ಗಮ್ಯ_apir_handle(name)

ಸ್ಥಿರ ಸಾಲೊಳಗೆ apir_ಬಫರ್_host_handle_t ggml_ಬಫರ್_ಗಮ್ಯ_apir_handle(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    // in the backend, the buffer handle is the buffer pointer
    ಹಿಂತಿರುಗಿ (apir_ಬಫರ್_host_handle_t) ಬಫರ್;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ apir_ಬಫರ್_type_host_handle_t ggml_ಬಫರ್_type_ಗಮ್ಯ_apir_handle(ggml_backend_ಬಫರ್_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    ಹಿಂತಿರುಗಿ (apir_ಬಫರ್_type_host_handle_t) buft;
}
