#समावेश "shared/apir_backend.h"

#घोषणा बफर_TO_HOST_HANDLE(name) ggml_बफर_to_apir_handle(name)

अटल अंतरभूत apir_बफर_host_handle_t ggml_बफर_to_apir_handle(ggml_backend_बफर_t बफर) {
    // in the backend, the buffer handle is the buffer pointer
    वापस (apir_बफर_host_handle_t) बफर;
}

अटल अंतरभूत apir_बफर_type_host_handle_t ggml_बफर_type_to_apir_handle(ggml_backend_बफर_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    वापस (apir_बफर_type_host_handle_t) buft;
}
