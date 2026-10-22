#ہدایت once

#شامل "ggml-backend-impl.h"
#شامل "ggml-backend.h"
#شامل "ggml-impl.h"
#شامل "virtgpu.h"

#شامل <memory>
#شامل <string>

#تعریف_کرو GGML_VIRTGPU_NAME "ggml-virtgpu"
#تعریف_کرو GGML_VIRTGPU      "ggml-virtgpu: "

// USE_ALWAYS_TRUE_SUPPORTS_OP: 1 is fast, 0 avoid micro-benchmark crashes

#تعریف_کرو USE_ALWAYS_TRUE_SUPPORTS_OP 1
#تعریف_کرو USE_METAL_GUEST_SUPPORTS_OP 0

#تعریف_کرو DEV_مقصد_GPU(name) ((ggml_backend_remoting_device_context *) (name)->context)->gpu

#تعریف_کرو بفر_مقصد_GGML_CONTEXT(name) ((ggml_backend_remoting_بفر_context *) (name)->context)

#تعریف_کرو بفر_مقصد_APIR_CONTEXT(name) &((ggml_backend_remoting_بفر_context *) (name)->context)->apir_context

#تعریف_کرو بفر_مقصد_HOST_HANDLE(name) ((ggml_backend_remoting_بفر_context *) (name)->context)->apir_context.host_handle

#تعریف_کرو GET_DEVICE_CONTEXT() (ggml_backend_remoting_device_context *) ggml_backend_remoting_get_device(0)->context

#تعریف_کرو BUFT_مقصد_GPU(name) ((ggml_backend_remoting_device_context *) (name)->device->context)->gpu

ساخت ggml_backend_remoting_device_context {
    size_t      device;
    std::string name;
    std::string description;

    std::سمتیہ<std::tuple<خالی *, size_t, virtgpu_shmem *>> shared_حافظہ;

    virtgpu * gpu;
};

ساخت ggml_backend_remoting_بفر_context {
    apir_بفر_context_t apir_context;

    virtgpu * gpu;

    خالی * بنیاد;

    bool is_from_ptr;
};

خارجی ثابت ggml_backend_بفر_type_i ggml_backend_remoting_بفر_type_interface;
خارجی ثابت ggml_backend_device_i      ggml_backend_remoting_device_interface;
خارجی ثابت ggml_backend_بفر_i      ggml_backend_remoting_بفر_interface;
خارجی ثابت ggml_backend_بفر_type_i ggml_backend_remoting_بفر_from_ptr_type_interface;
خارجی ثابت ggml_backend_بفر_i      ggml_backend_remoting_بفر_from_ptr_interface;

ggml_backend_dev_t         ggml_backend_remoting_get_device(size_t device);
ggml_backend_t             ggml_backend_remoting_device_init(ggml_backend_dev_t dev, ثابت حرف * params);
ggml_backend_بفر_type_t ggml_backend_remoting_device_get_بفر_type(ggml_backend_dev_t dev);

ساکن اندرونی apir_بفر_type_host_handle_t ggml_بفر_type_مقصد_apir_handle(ggml_backend_بفر_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    واپس (apir_بفر_type_host_handle_t) buft->context;
}

ساکن اندرونی apir_بفر_host_handle_t ggml_بفر_مقصد_apir_handle(ggml_backend_بفر_t بفر) {
    اگر (!بفر->context) {
        GGML_ABORT(GGML_VIRTGPU "%s: no context available :/", __func__);
    }
    واپس بفر_مقصد_HOST_HANDLE(بفر);
}
