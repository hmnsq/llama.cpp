#هدايت once

#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-impl.h"
#ضمن "virtgpu.h"

#ضمن <memory>
#ضمن <string>

#وصف_ڪريو GGML_VIRTGPU_NAME "ggml-virtgpu"
#وصف_ڪريو GGML_VIRTGPU      "ggml-virtgpu: "

// USE_ALWAYS_TRUE_SUPPORTS_OP: 1 is fast, 0 avoid micro-benchmark crashes

#وصف_ڪريو USE_ALWAYS_TRUE_SUPPORTS_OP 1
#وصف_ڪريو USE_METAL_GUEST_SUPPORTS_OP 0

#وصف_ڪريو DEV_منزل_GPU(name) ((ggml_backend_remoting_device_context *) (name)->context)->gpu

#وصف_ڪريو بفر_منزل_GGML_CONTEXT(name) ((ggml_backend_remoting_بفر_context *) (name)->context)

#وصف_ڪريو بفر_منزل_APIR_CONTEXT(name) &((ggml_backend_remoting_بفر_context *) (name)->context)->apir_context

#وصف_ڪريو بفر_منزل_HOST_HANDLE(name) ((ggml_backend_remoting_بفر_context *) (name)->context)->apir_context.host_handle

#وصف_ڪريو GET_DEVICE_CONTEXT() (ggml_backend_remoting_device_context *) ggml_backend_remoting_get_device(0)->context

#وصف_ڪريو BUFT_منزل_GPU(name) ((ggml_backend_remoting_device_context *) (name)->device->context)->gpu

بنية ggml_backend_remoting_device_context {
    size_t      device;
    std::string name;
    std::string description;

    std::ويڪٽر<std::tuple<فراغ *, size_t, virtgpu_shmem *>> shared_يادگيري;

    virtgpu * gpu;
};

بنية ggml_backend_remoting_بفر_context {
    apir_بفر_context_t apir_context;

    virtgpu * gpu;

    فراغ * بنياد;

    bool is_from_ptr;
};

خارجي ثابت ggml_backend_بفر_type_i ggml_backend_remoting_بفر_type_interface;
خارجي ثابت ggml_backend_device_i      ggml_backend_remoting_device_interface;
خارجي ثابت ggml_backend_بفر_i      ggml_backend_remoting_بفر_interface;
خارجي ثابت ggml_backend_بفر_type_i ggml_backend_remoting_بفر_from_ptr_type_interface;
خارجي ثابت ggml_backend_بفر_i      ggml_backend_remoting_بفر_from_ptr_interface;

ggml_backend_dev_t         ggml_backend_remoting_get_device(size_t device);
ggml_backend_t             ggml_backend_remoting_device_init(ggml_backend_dev_t dev, ثابت حرف * params);
ggml_backend_بفر_type_t ggml_backend_remoting_device_get_بفر_type(ggml_backend_dev_t dev);

ساكن مضمن apir_بفر_type_host_handle_t ggml_بفر_type_منزل_apir_handle(ggml_backend_بفر_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    أرجع (apir_بفر_type_host_handle_t) buft->context;
}

ساكن مضمن apir_بفر_host_handle_t ggml_بفر_منزل_apir_handle(ggml_backend_بفر_t بفر) {
    إذا (!بفر->context) {
        GGML_ABORT(GGML_VIRTGPU "%s: no context available :/", __func__);
    }
    أرجع بفر_منزل_HOST_HANDLE(بفر);
}
