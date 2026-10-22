#अन्तर्भाव "shared/apir_backend.h"

#परिभाषय बफर_गन्तव्यम्_HOST_HANDLE(name) ggml_बफर_गन्तव्यम्_apir_handle(name)

स्थिर अन्तःस्थ apir_बफर_host_handle_t ggml_बफर_गन्तव्यम्_apir_handle(ggml_backend_बफर_t बफर) {
    // in the backend, the buffer handle is the buffer pointer
    प्रत्यागम (apir_बफर_host_handle_t) बफर;
}

स्थिर अन्तःस्थ apir_बफर_type_host_handle_t ggml_बफर_type_गन्तव्यम्_apir_handle(ggml_backend_बफर_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    प्रत्यागम (apir_बफर_type_host_handle_t) buft;
}
