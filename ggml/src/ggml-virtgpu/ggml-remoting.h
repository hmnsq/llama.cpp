#निर्देशः once

#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "virtgpu.h"

#अन्तर्भाव <memory>
#अन्तर्भाव <string>

#परिभाषय GGML_VIRTGPU_NAME "ggml-virtgpu"
#परिभाषय GGML_VIRTGPU      "ggml-virtgpu: "

// USE_ALWAYS_TRUE_SUPPORTS_OP: 1 is fast, 0 avoid micro-benchmark crashes

#परिभाषय USE_ALWAYS_TRUE_SUPPORTS_OP 1
#परिभाषय USE_METAL_GUEST_SUPPORTS_OP 0

#परिभाषय DEV_गन्तव्यम्_GPU(name) ((ggml_backend_remoting_device_context *) (name)->context)->gpu

#परिभाषय बफर_गन्तव्यम्_GGML_CONTEXT(name) ((ggml_backend_remoting_बफर_context *) (name)->context)

#परिभाषय बफर_गन्तव्यम्_APIR_CONTEXT(name) &((ggml_backend_remoting_बफर_context *) (name)->context)->apir_context

#परिभाषय बफर_गन्तव्यम्_HOST_HANDLE(name) ((ggml_backend_remoting_बफर_context *) (name)->context)->apir_context.host_handle

#परिभाषय GET_DEVICE_CONTEXT() (ggml_backend_remoting_device_context *) ggml_backend_remoting_get_device(0)->context

#परिभाषय BUFT_गन्तव्यम्_GPU(name) ((ggml_backend_remoting_device_context *) (name)->device->context)->gpu

संरचना ggml_backend_remoting_device_context {
    size_t      device;
    std::string name;
    std::string description;

    std::सदिशम्<std::tuple<शून्य *, size_t, virtgpu_shmem *>> shared_स्मृतिः;

    virtgpu * gpu;
};

संरचना ggml_backend_remoting_बफर_context {
    apir_बफर_context_t apir_context;

    virtgpu * gpu;

    शून्य * आधारः;

    bool is_from_ptr;
};

बाह्य ध्रुव ggml_backend_बफर_type_i ggml_backend_remoting_बफर_type_interface;
बाह्य ध्रुव ggml_backend_device_i      ggml_backend_remoting_device_interface;
बाह्य ध्रुव ggml_backend_बफर_i      ggml_backend_remoting_बफर_interface;
बाह्य ध्रुव ggml_backend_बफर_type_i ggml_backend_remoting_बफर_from_ptr_type_interface;
बाह्य ध्रुव ggml_backend_बफर_i      ggml_backend_remoting_बफर_from_ptr_interface;

ggml_backend_dev_t         ggml_backend_remoting_get_device(size_t device);
ggml_backend_t             ggml_backend_remoting_device_init(ggml_backend_dev_t dev, ध्रुव वर्ण * params);
ggml_backend_बफर_type_t ggml_backend_remoting_device_get_बफर_type(ggml_backend_dev_t dev);

स्थिर अन्तःस्थ apir_बफर_type_host_handle_t ggml_बफर_type_गन्तव्यम्_apir_handle(ggml_backend_बफर_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    प्रत्यागम (apir_बफर_type_host_handle_t) buft->context;
}

स्थिर अन्तःस्थ apir_बफर_host_handle_t ggml_बफर_गन्तव्यम्_apir_handle(ggml_backend_बफर_t बफर) {
    यदि (!बफर->context) {
        GGML_ABORT(GGML_VIRTGPU "%s: no context available :/", __func__);
    }
    प्रत्यागम बफर_गन्तव्यम्_HOST_HANDLE(बफर);
}
