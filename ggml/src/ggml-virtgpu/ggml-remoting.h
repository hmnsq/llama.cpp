#pragma once

#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-backend.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "virtgpu.h"

#অন্তর্ভাব <memory>
#অন্তর্ভাব <string>

#define GGML_VIRTGPU_NAME "ggml-virtgpu"
#define GGML_VIRTGPU      "ggml-virtgpu: "

// USE_ALWAYS_TRUE_SUPPORTS_OP: 1 is fast, 0 avoid micro-benchmark crashes

#define USE_ALWAYS_TRUE_SUPPORTS_OP 1
#define USE_METAL_GUEST_SUPPORTS_OP 0

#define DEV_TO_GPU(name) ((ggml_backend_remoting_device_context *) (name)->context)->gpu

#define বাফার_TO_GGML_CONTEXT(name) ((ggml_backend_remoting_বাফার_context *) (name)->context)

#define বাফার_TO_APIR_CONTEXT(name) &((ggml_backend_remoting_বাফার_context *) (name)->context)->apir_context

#define বাফার_TO_HOST_HANDLE(name) ((ggml_backend_remoting_বাফার_context *) (name)->context)->apir_context.host_handle

#define GET_DEVICE_CONTEXT() (ggml_backend_remoting_device_context *) ggml_backend_remoting_get_device(0)->context

#define BUFT_TO_GPU(name) ((ggml_backend_remoting_device_context *) (name)->device->context)->gpu

কাঠামো ggml_backend_remoting_device_context {
    size_t      device;
    std::string name;
    std::string description;

    std::vector<std::tuple<শূন্য *, size_t, virtgpu_shmem *>> shared_memory;

    virtgpu * gpu;
};

কাঠামো ggml_backend_remoting_বাফার_context {
    apir_বাফার_context_t apir_context;

    virtgpu * gpu;

    শূন্য * base;

    bool is_from_ptr;
};

বাহ্যিক ধ্রুবক ggml_backend_বাফার_type_i ggml_backend_remoting_বাফার_type_interface;
বাহ্যিক ধ্রুবক ggml_backend_device_i      ggml_backend_remoting_device_interface;
বাহ্যিক ধ্রুবক ggml_backend_বাফার_i      ggml_backend_remoting_বাফার_interface;
বাহ্যিক ধ্রুবক ggml_backend_বাফার_type_i ggml_backend_remoting_বাফার_from_ptr_type_interface;
বাহ্যিক ধ্রুবক ggml_backend_বাফার_i      ggml_backend_remoting_বাফার_from_ptr_interface;

ggml_backend_dev_t         ggml_backend_remoting_get_device(size_t device);
ggml_backend_t             ggml_backend_remoting_device_init(ggml_backend_dev_t dev, ধ্রুবক অক্ষর * params);
ggml_backend_বাফার_type_t ggml_backend_remoting_device_get_বাফার_type(ggml_backend_dev_t dev);

স্থির অন্তঃরেখ apir_বাফার_type_host_handle_t ggml_বাফার_type_to_apir_handle(ggml_backend_বাফার_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    ফেরত (apir_বাফার_type_host_handle_t) buft->context;
}

স্থির অন্তঃরেখ apir_বাফার_host_handle_t ggml_বাফার_to_apir_handle(ggml_backend_বাফার_t বাফার) {
    যদি (!বাফার->context) {
        GGML_ABORT(GGML_VIRTGPU "%s: no context available :/", __func__);
    }
    ফেরত বাফার_TO_HOST_HANDLE(বাফার);
}
