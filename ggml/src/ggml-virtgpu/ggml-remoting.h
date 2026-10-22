#رهنمود once

#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-impl.h"
#ضمن "virtgpu.h"

#ضمن <memory>
#ضمن <string>

#تعریف_کن GGML_VIRTGPU_NAME "ggml-virtgpu"
#تعریف_کن GGML_VIRTGPU      "ggml-virtgpu: "

// USE_ALWAYS_TRUE_SUPPORTS_OP: 1 is fast, 0 avoid micro-benchmark crashes

#تعریف_کن USE_ALWAYS_TRUE_SUPPORTS_OP 1
#تعریف_کن USE_METAL_GUEST_SUPPORTS_OP 0

#تعریف_کن DEV_مقصد_GPU(name) ((ggml_backend_remoting_device_context *) (name)->context)->gpu

#تعریف_کن میانگیر_مقصد_GGML_CONTEXT(name) ((ggml_backend_remoting_میانگیر_context *) (name)->context)

#تعریف_کن میانگیر_مقصد_APIR_CONTEXT(name) &((ggml_backend_remoting_میانگیر_context *) (name)->context)->apir_context

#تعریف_کن میانگیر_مقصد_HOST_HANDLE(name) ((ggml_backend_remoting_میانگیر_context *) (name)->context)->apir_context.host_handle

#تعریف_کن GET_DEVICE_CONTEXT() (ggml_backend_remoting_device_context *) ggml_backend_remoting_get_device(0)->context

#تعریف_کن BUFT_مقصد_GPU(name) ((ggml_backend_remoting_device_context *) (name)->device->context)->gpu

بنية ggml_backend_remoting_device_context {
    size_t      device;
    std::string name;
    std::string description;

    std::وکتور<std::tuple<فراغ *, size_t, virtgpu_shmem *>> shared_حافظه;

    virtgpu * gpu;
};

بنية ggml_backend_remoting_میانگیر_context {
    apir_میانگیر_context_t apir_context;

    virtgpu * gpu;

    فراغ * پایه;

    bool is_from_ptr;
};

خارجي ثابت ggml_backend_میانگیر_type_i ggml_backend_remoting_میانگیر_type_interface;
خارجي ثابت ggml_backend_device_i      ggml_backend_remoting_device_interface;
خارجي ثابت ggml_backend_میانگیر_i      ggml_backend_remoting_میانگیر_interface;
خارجي ثابت ggml_backend_میانگیر_type_i ggml_backend_remoting_میانگیر_from_ptr_type_interface;
خارجي ثابت ggml_backend_میانگیر_i      ggml_backend_remoting_میانگیر_from_ptr_interface;

ggml_backend_dev_t         ggml_backend_remoting_get_device(size_t device);
ggml_backend_t             ggml_backend_remoting_device_init(ggml_backend_dev_t dev, ثابت حرف * params);
ggml_backend_میانگیر_type_t ggml_backend_remoting_device_get_میانگیر_type(ggml_backend_dev_t dev);

ساكن مضمن apir_میانگیر_type_host_handle_t ggml_میانگیر_type_مقصد_apir_handle(ggml_backend_میانگیر_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    أرجع (apir_میانگیر_type_host_handle_t) buft->context;
}

ساكن مضمن apir_میانگیر_host_handle_t ggml_میانگیر_مقصد_apir_handle(ggml_backend_میانگیر_t میانگیر) {
    إذا (!میانگیر->context) {
        GGML_ABORT(GGML_VIRTGPU "%s: no context available :/", __func__);
    }
    أرجع میانگیر_مقصد_HOST_HANDLE(میانگیر);
}
