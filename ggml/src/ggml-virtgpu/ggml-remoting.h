#आशय once

#समावेश "ggml-backend-impl.h"
#समावेश "ggml-backend.h"
#समावेश "ggml-impl.h"
#समावेश "virtgpu.h"

#समावेश <memory>
#समावेश <string>

#घोषणा GGML_VIRTGPU_NAME "ggml-virtgpu"
#घोषणा GGML_VIRTGPU      "ggml-virtgpu: "

// USE_ALWAYS_TRUE_SUPPORTS_OP: 1 is fast, 0 avoid micro-benchmark crashes

#घोषणा USE_ALWAYS_TRUE_SUPPORTS_OP 1
#घोषणा USE_METAL_GUEST_SUPPORTS_OP 0

#घोषणा DEV_TO_GPU(name) ((ggml_backend_remoting_device_context *) (name)->context)->gpu

#घोषणा बफर_TO_GGML_CONTEXT(name) ((ggml_backend_remoting_बफर_context *) (name)->context)

#घोषणा बफर_TO_APIR_CONTEXT(name) &((ggml_backend_remoting_बफर_context *) (name)->context)->apir_context

#घोषणा बफर_TO_HOST_HANDLE(name) ((ggml_backend_remoting_बफर_context *) (name)->context)->apir_context.host_handle

#घोषणा GET_DEVICE_CONTEXT() (ggml_backend_remoting_device_context *) ggml_backend_remoting_get_device(0)->context

#घोषणा BUFT_TO_GPU(name) ((ggml_backend_remoting_device_context *) (name)->device->context)->gpu

काष्ठा ggml_backend_remoting_device_context {
    माप_प्रकार      device;
    std::string name;
    std::string description;

    std::vector<std::tuple<व्योम *, माप_प्रकार, virtgpu_shmem *>> shared_memory;

    virtgpu * gpu;
};

काष्ठा ggml_backend_remoting_बफर_context {
    apir_बफर_context_t apir_context;

    virtgpu * gpu;

    व्योम * base;

    bool is_from_ptr;
};

बाह्य स्थिर ggml_backend_बफर_type_i ggml_backend_remoting_बफर_type_interface;
बाह्य स्थिर ggml_backend_device_i      ggml_backend_remoting_device_interface;
बाह्य स्थिर ggml_backend_बफर_i      ggml_backend_remoting_बफर_interface;
बाह्य स्थिर ggml_backend_बफर_type_i ggml_backend_remoting_बफर_from_ptr_type_interface;
बाह्य स्थिर ggml_backend_बफर_i      ggml_backend_remoting_बफर_from_ptr_interface;

ggml_backend_dev_t         ggml_backend_remoting_get_device(माप_प्रकार device);
ggml_backend_t             ggml_backend_remoting_device_init(ggml_backend_dev_t dev, स्थिर अक्षर * params);
ggml_backend_बफर_type_t ggml_backend_remoting_device_get_बफर_type(ggml_backend_dev_t dev);

अटल अंतरभूत apir_बफर_type_host_handle_t ggml_बफर_type_to_apir_handle(ggml_backend_बफर_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    वापस (apir_बफर_type_host_handle_t) buft->context;
}

अटल अंतरभूत apir_बफर_host_handle_t ggml_बफर_to_apir_handle(ggml_backend_बफर_t बफर) {
    अगर (!बफर->context) {
        GGML_ABORT(GGML_VIRTGPU "%s: no context available :/", __func__);
    }
    वापस बफर_TO_HOST_HANDLE(बफर);
}
