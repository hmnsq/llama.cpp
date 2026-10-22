#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "virtgpu.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_VIRTGPU_NAME "ggml-virtgpu"
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_VIRTGPU      "ggml-virtgpu: "

// USE_ALWAYS_TRUE_SUPPORTS_OP: 1 is fast, 0 avoid micro-benchmark crashes

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ USE_ALWAYS_TRUE_SUPPORTS_OP 1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ USE_METAL_GUEST_SUPPORTS_OP 0

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DEV_ਮੰਜ਼ਿਲ_GPU(name) ((ggml_backend_remoting_device_context *) (name)->context)->gpu

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਬਫਰ_ਮੰਜ਼ਿਲ_GGML_CONTEXT(name) ((ggml_backend_remoting_ਬਫਰ_context *) (name)->context)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਬਫਰ_ਮੰਜ਼ਿਲ_APIR_CONTEXT(name) &((ggml_backend_remoting_ਬਫਰ_context *) (name)->context)->apir_context

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਬਫਰ_ਮੰਜ਼ਿਲ_HOST_HANDLE(name) ((ggml_backend_remoting_ਬਫਰ_context *) (name)->context)->apir_context.host_handle

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GET_DEVICE_CONTEXT() (ggml_backend_remoting_device_context *) ggml_backend_remoting_get_device(0)->context

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ BUFT_ਮੰਜ਼ਿਲ_GPU(name) ((ggml_backend_remoting_device_context *) (name)->device->context)->gpu

ਬਣਤਰ ggml_backend_remoting_device_context {
    size_t      device;
    std::string name;
    std::string description;

    std::ਸਦਿਸ਼<std::tuple<ਖਾਲੀ *, size_t, virtgpu_shmem *>> shared_ਸਮ੍ਰਿਤੀ;

    virtgpu * gpu;
};

ਬਣਤਰ ggml_backend_remoting_ਬਫਰ_context {
    apir_ਬਫਰ_context_t apir_context;

    virtgpu * gpu;

    ਖਾਲੀ * ਅਧਾਰ;

    bool is_from_ptr;
};

ਬਾਹਰੀ ਅਚਲ ggml_backend_ਬਫਰ_type_i ggml_backend_remoting_ਬਫਰ_type_interface;
ਬਾਹਰੀ ਅਚਲ ggml_backend_device_i      ggml_backend_remoting_device_interface;
ਬਾਹਰੀ ਅਚਲ ggml_backend_ਬਫਰ_i      ggml_backend_remoting_ਬਫਰ_interface;
ਬਾਹਰੀ ਅਚਲ ggml_backend_ਬਫਰ_type_i ggml_backend_remoting_ਬਫਰ_from_ptr_type_interface;
ਬਾਹਰੀ ਅਚਲ ggml_backend_ਬਫਰ_i      ggml_backend_remoting_ਬਫਰ_from_ptr_interface;

ggml_backend_dev_t         ggml_backend_remoting_get_device(size_t device);
ggml_backend_t             ggml_backend_remoting_device_init(ggml_backend_dev_t dev, ਅਚਲ ਅੱਖਰ * params);
ggml_backend_ਬਫਰ_type_t ggml_backend_remoting_device_get_ਬਫਰ_type(ggml_backend_dev_t dev);

ਸਥਿਰ ਸਤਰਅੰਦਰ apir_ਬਫਰ_type_host_handle_t ggml_ਬਫਰ_type_ਮੰਜ਼ਿਲ_apir_handle(ggml_backend_ਬਫਰ_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    ਵਾਪਸ (apir_ਬਫਰ_type_host_handle_t) buft->context;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ apir_ਬਫਰ_host_handle_t ggml_ਬਫਰ_ਮੰਜ਼ਿਲ_apir_handle(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ਜੇ (!ਬਫਰ->context) {
        GGML_ABORT(GGML_VIRTGPU "%s: no context available :/", __func__);
    }
    ਵਾਪਸ ਬਫਰ_ਮੰਜ਼ਿਲ_HOST_HANDLE(ਬਫਰ);
}
