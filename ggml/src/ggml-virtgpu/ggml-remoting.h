#pragma once

#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-backend.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "virtgpu.h"

#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <string>

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

સંરચના ggml_backend_remoting_device_context {
    size_t      device;
    std::string name;
    std::string description;

    std::vector<std::tuple<શૂન્ય *, size_t, virtgpu_shmem *>> shared_memory;

    virtgpu * gpu;
};

સંરચના ggml_backend_remoting_buffer_context {
    apir_buffer_context_t apir_context;

    virtgpu * gpu;

    શૂન્ય * base;

    bool is_from_ptr;
};

બાહ્ય અચલ ggml_backend_buffer_type_i ggml_backend_remoting_buffer_type_interface;
બાહ્ય અચલ ggml_backend_device_i      ggml_backend_remoting_device_interface;
બાહ્ય અચલ ggml_backend_buffer_i      ggml_backend_remoting_buffer_interface;
બાહ્ય અચલ ggml_backend_buffer_type_i ggml_backend_remoting_buffer_from_ptr_type_interface;
બાહ્ય અચલ ggml_backend_buffer_i      ggml_backend_remoting_buffer_from_ptr_interface;

ggml_backend_dev_t         ggml_backend_remoting_get_device(size_t device);
ggml_backend_t             ggml_backend_remoting_device_init(ggml_backend_dev_t dev, અચલ અક્ષર * params);
ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_type(ggml_backend_dev_t dev);

સ્થિર અંતઃરેખ apir_buffer_type_host_handle_t ggml_buffer_type_to_apir_handle(ggml_backend_buffer_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    પાછા (apir_buffer_type_host_handle_t) buft->context;
}

સ્થિર અંતઃરેખ apir_buffer_host_handle_t ggml_buffer_to_apir_handle(ggml_backend_buffer_t buffer) {
    જો (!buffer->context) {
        GGML_ABORT(GGML_VIRTGPU "%s: no context available :/", __func__);
    }
    પાછા BUFFER_TO_HOST_HANDLE(buffer);
}
