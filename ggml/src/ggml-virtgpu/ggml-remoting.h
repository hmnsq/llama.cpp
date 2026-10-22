#pragma once

#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-backend.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "virtgpu.h"

#அந்தர்பாவ <memory>
#அந்தர்பாவ <string>

#define GGML_VIRTGPU_NAME "ggml-virtgpu"
#define GGML_VIRTGPU      "ggml-virtgpu: "

// USE_ALWAYS_TRUE_SUPPORTS_OP: 1 is fast, 0 avoid micro-benchmark crashes

#define USE_ALWAYS_TRUE_SUPPORTS_OP 1
#define USE_METAL_GUEST_SUPPORTS_OP 0

#define DEV_TO_GPU(name) ((ggml_backend_remoting_device_context *) (name)->context)->gpu

#define BUFFER_TO_GGML_CONTEXT(name) ((ggml_backend_remoting_buffer_context *) (name)->context)

#define BUFFER_TO_APIR_CONTEXT(name) &((ggml_backend_remoting_buffer_context *) (name)->context)->apir_context

#define BUFFER_TO_HOST_HANDLE(name) ((ggml_backend_remoting_buffer_context *) (name)->context)->apir_context.host_handle

#define GET_DEVICE_CONTEXT() (ggml_backend_remoting_device_context *) ggml_backend_remoting_get_device(0)->context

#define BUFT_TO_GPU(name) ((ggml_backend_remoting_device_context *) (name)->device->context)->gpu

கட்டமைப்பு ggml_backend_remoting_device_context {
    size_t      device;
    std::string name;
    std::string description;

    std::vector<std::tuple<வெறுமை *, size_t, virtgpu_shmem *>> shared_memory;

    virtgpu * gpu;
};

கட்டமைப்பு ggml_backend_remoting_buffer_context {
    apir_buffer_context_t apir_context;

    virtgpu * gpu;

    வெறுமை * base;

    bool is_from_ptr;
};

வெளி மாறிலி ggml_backend_buffer_type_i ggml_backend_remoting_buffer_type_interface;
வெளி மாறிலி ggml_backend_device_i      ggml_backend_remoting_device_interface;
வெளி மாறிலி ggml_backend_buffer_i      ggml_backend_remoting_buffer_interface;
வெளி மாறிலி ggml_backend_buffer_type_i ggml_backend_remoting_buffer_from_ptr_type_interface;
வெளி மாறிலி ggml_backend_buffer_i      ggml_backend_remoting_buffer_from_ptr_interface;

ggml_backend_dev_t         ggml_backend_remoting_get_device(size_t device);
ggml_backend_t             ggml_backend_remoting_device_init(ggml_backend_dev_t dev, மாறிலி எழுத்து * params);
ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_type(ggml_backend_dev_t dev);

நிலையான உள்வரி apir_buffer_type_host_handle_t ggml_buffer_type_to_apir_handle(ggml_backend_buffer_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    திரும்பு (apir_buffer_type_host_handle_t) buft->context;
}

நிலையான உள்வரி apir_buffer_host_handle_t ggml_buffer_to_apir_handle(ggml_backend_buffer_t buffer) {
    எனில் (!buffer->context) {
        GGML_ABORT(GGML_VIRTGPU "%s: no context available :/", __func__);
    }
    திரும்பு BUFFER_TO_HOST_HANDLE(buffer);
}
