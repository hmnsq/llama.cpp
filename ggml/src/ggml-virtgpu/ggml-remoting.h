#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "virtgpu.h"

#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <string>

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_VIRTGPU_NAME "ggml-virtgpu"
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_VIRTGPU      "ggml-virtgpu: "

// USE_ALWAYS_TRUE_SUPPORTS_OP: 1 is fast, 0 avoid micro-benchmark crashes

#ವ್ಯಾಖ್ಯಾನಿಸು USE_ALWAYS_TRUE_SUPPORTS_OP 1
#ವ್ಯಾಖ್ಯಾನಿಸು USE_METAL_GUEST_SUPPORTS_OP 0

#ವ್ಯಾಖ್ಯಾನಿಸು DEV_ಗಮ್ಯ_GPU(name) ((ggml_backend_remoting_device_context *) (name)->context)->gpu

#ವ್ಯಾಖ್ಯಾನಿಸು ಬಫರ್_ಗಮ್ಯ_GGML_CONTEXT(name) ((ggml_backend_remoting_ಬಫರ್_context *) (name)->context)

#ವ್ಯಾಖ್ಯಾನಿಸು ಬಫರ್_ಗಮ್ಯ_APIR_CONTEXT(name) &((ggml_backend_remoting_ಬಫರ್_context *) (name)->context)->apir_context

#ವ್ಯಾಖ್ಯಾನಿಸು ಬಫರ್_ಗಮ್ಯ_HOST_HANDLE(name) ((ggml_backend_remoting_ಬಫರ್_context *) (name)->context)->apir_context.host_handle

#ವ್ಯಾಖ್ಯಾನಿಸು GET_DEVICE_CONTEXT() (ggml_backend_remoting_device_context *) ggml_backend_remoting_get_device(0)->context

#ವ್ಯಾಖ್ಯಾನಿಸು BUFT_ಗಮ್ಯ_GPU(name) ((ggml_backend_remoting_device_context *) (name)->device->context)->gpu

ರಚನೆ ggml_backend_remoting_device_context {
    size_t      device;
    std::string name;
    std::string description;

    std::ಸದಿಶ<std::tuple<ಶೂನ್ಯ *, size_t, virtgpu_shmem *>> shared_ಸ್ಮರಣೆ;

    virtgpu * gpu;
};

ರಚನೆ ggml_backend_remoting_ಬಫರ್_context {
    apir_ಬಫರ್_context_t apir_context;

    virtgpu * gpu;

    ಶೂನ್ಯ * ತಳ;

    bool is_from_ptr;
};

ಬಾಹ್ಯ ಸ್ಥಿರಾಂಕ ggml_backend_ಬಫರ್_type_i ggml_backend_remoting_ಬಫರ್_type_interface;
ಬಾಹ್ಯ ಸ್ಥಿರಾಂಕ ggml_backend_device_i      ggml_backend_remoting_device_interface;
ಬಾಹ್ಯ ಸ್ಥಿರಾಂಕ ggml_backend_ಬಫರ್_i      ggml_backend_remoting_ಬಫರ್_interface;
ಬಾಹ್ಯ ಸ್ಥಿರಾಂಕ ggml_backend_ಬಫರ್_type_i ggml_backend_remoting_ಬಫರ್_from_ptr_type_interface;
ಬಾಹ್ಯ ಸ್ಥಿರಾಂಕ ggml_backend_ಬಫರ್_i      ggml_backend_remoting_ಬಫರ್_from_ptr_interface;

ggml_backend_dev_t         ggml_backend_remoting_get_device(size_t device);
ggml_backend_t             ggml_backend_remoting_device_init(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params);
ggml_backend_ಬಫರ್_type_t ggml_backend_remoting_device_get_ಬಫರ್_type(ggml_backend_dev_t dev);

ಸ್ಥಿರ ಸಾಲೊಳಗೆ apir_ಬಫರ್_type_host_handle_t ggml_ಬಫರ್_type_ಗಮ್ಯ_apir_handle(ggml_backend_ಬಫರ್_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    ಹಿಂತಿರುಗಿ (apir_ಬಫರ್_type_host_handle_t) buft->context;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ apir_ಬಫರ್_host_handle_t ggml_ಬಫರ್_ಗಮ್ಯ_apir_handle(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ಆದರೆ (!ಬಫರ್->context) {
        GGML_ABORT(GGML_VIRTGPU "%s: no context available :/", __func__);
    }
    ಹಿಂತಿರುಗಿ ಬಫರ್_ಗಮ್ಯ_HOST_HANDLE(ಬಫರ್);
}
