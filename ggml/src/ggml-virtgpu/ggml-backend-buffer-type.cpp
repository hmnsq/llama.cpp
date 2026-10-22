#ضمن "ggml-remoting.h"

ساكن ggml_backend_میانگیر_t ggml_backend_remoting_میانگیر_type_alloc_میانگیر(ggml_backend_میانگیر_type_t buft,
                                                                            size_t                     size) {
    virtgpu * gpu = BUFT_مقصد_GPU(buft);

    ggml_backend_remoting_میانگیر_context * context = (ggml_backend_remoting_میانگیر_context *) حافظه_بده(حجم(*context));
    إذا (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu = gpu;

    منطقي async_استفادهنشده, host_میانگیر_استفادهنشده, events_استفادهنشده, mmap_support_استفادهنشده;
    منطقي میانگیر_from_host_ptr;
    apir_device_get_props(gpu, &async_استفادهنشده, &host_میانگیر_استفادهنشده, &میانگیر_from_host_ptr, &events_استفادهنشده, &mmap_support_استفادهنشده);

    إذا (میانگیر_from_host_ptr) {
        context->apir_context = apir_device_میانگیر_from_ptr(gpu, size, size);
        context->پایه         = context->apir_context.shmem.mmap_ptr;
        context->is_from_ptr  = درست;
    } وإلا {
        context->apir_context = apir_میانگیر_type_alloc_میانگیر(gpu, gpu->cached_میانگیر_type.host_handle, size);
        context->is_from_ptr  = خطأ;
        context->پایه         = عدم;
    }

    ggml_backend_میانگیر_t میانگیر =
        ggml_backend_میانگیر_init(buft, ggml_backend_remoting_میانگیر_interface, (فراغ *) context, size);

    أرجع میانگیر;
}

ساكن ثابت حرف * ggml_backend_remoting_میانگیر_type_get_name(ggml_backend_میانگیر_type_t buft) {
    virtgpu * gpu = BUFT_مقصد_GPU(buft);

    // Return the prefixed name that was built once during initialization
    أرجع gpu->cached_میانگیر_type.name;
}

ساكن size_t ggml_backend_remoting_میانگیر_type_get_alignment(ggml_backend_میانگیر_type_t buft) {
    virtgpu * gpu = BUFT_مقصد_GPU(buft);

    أرجع gpu->cached_میانگیر_type.alignment;
}

ساكن size_t ggml_backend_remoting_میانگیر_type_get_max_اندازه(ggml_backend_میانگیر_type_t buft) {
    virtgpu * gpu = BUFT_مقصد_GPU(buft);

    أرجع gpu->cached_میانگیر_type.max_size;
}

ساكن size_t ggml_backend_remoting_میانگیر_type_get_alloc_اندازه(ggml_backend_میانگیر_type_t buft,
                                                               ثابت ggml_tensor *        tensor) {
    virtgpu * gpu = BUFT_مقصد_GPU(buft);

    إذا (tensor->میانگیر == عدم || !tensor->میانگیر->context ||
        !buft->device->iface.supports_buft(buft->device, tensor->میانگیر->buft)) {
        أرجع ggml_nbytes(tensor);
    }

    أرجع apir_میانگیر_type_get_alloc_اندازه(gpu, gpu->cached_میانگیر_type.host_handle, tensor);
}

ثابت ggml_backend_میانگیر_type_i ggml_backend_remoting_میانگیر_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_میانگیر_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_remoting_میانگیر_type_alloc_میانگیر,
    /* .alloc_buffer_n      = */ عدم,
    /* .get_alignment       = */ ggml_backend_remoting_میانگیر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_میانگیر_type_get_max_اندازه,
    /* .get_alloc_size      = */ ggml_backend_remoting_میانگیر_type_get_alloc_اندازه,
    /* .get_alloc_size_n    = */ عدم,
    /* .is_host             = */ عدم,
};

ثابت ggml_backend_میانگیر_type_i ggml_backend_remoting_میانگیر_from_ptr_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_میانگیر_type_get_name,
    /* .alloc_buffer        = */ عدم,
    /* .alloc_buffer_n      = */ عدم,
    /* .get_alignment       = */ ggml_backend_remoting_میانگیر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_میانگیر_type_get_max_اندازه,
    /* .get_alloc_size      = */ ggml_backend_remoting_میانگیر_type_get_alloc_اندازه,
    /* .get_alloc_size_n    = */ عدم,
    /* .is_host             = */ عدم,
};
