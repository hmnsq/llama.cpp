#شامل "ggml-remoting.h"

ساکن ggml_backend_بفر_t ggml_backend_remoting_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft,
                                                                            size_t                     size) {
    virtgpu * gpu = BUFT_مقصد_GPU(buft);

    ggml_backend_remoting_بفر_context * context = (ggml_backend_remoting_بفر_context *) حافظہ_دو(حجم(*context));
    اگر (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu = gpu;

    منطقی async_غیر_مستعمل, host_بفر_غیر_مستعمل, events_غیر_مستعمل, mmap_support_غیر_مستعمل;
    منطقی بفر_from_host_ptr;
    apir_device_get_props(gpu, &async_غیر_مستعمل, &host_بفر_غیر_مستعمل, &بفر_from_host_ptr, &events_غیر_مستعمل, &mmap_support_غیر_مستعمل);

    اگر (بفر_from_host_ptr) {
        context->apir_context = apir_device_بفر_from_ptr(gpu, size, size);
        context->بنیاد         = context->apir_context.shmem.mmap_ptr;
        context->is_from_ptr  = سچ;
    } ورنہ {
        context->apir_context = apir_بفر_type_alloc_بفر(gpu, gpu->cached_بفر_type.host_handle, size);
        context->is_from_ptr  = جھوٹ;
        context->بنیاد         = تہی;
    }

    ggml_backend_بفر_t بفر =
        ggml_backend_بفر_init(buft, ggml_backend_remoting_بفر_interface, (خالی *) context, size);

    واپس بفر;
}

ساکن ثابت حرف * ggml_backend_remoting_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    virtgpu * gpu = BUFT_مقصد_GPU(buft);

    // Return the prefixed name that was built once during initialization
    واپس gpu->cached_بفر_type.name;
}

ساکن size_t ggml_backend_remoting_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    virtgpu * gpu = BUFT_مقصد_GPU(buft);

    واپس gpu->cached_بفر_type.alignment;
}

ساکن size_t ggml_backend_remoting_بفر_type_get_max_سائز(ggml_backend_بفر_type_t buft) {
    virtgpu * gpu = BUFT_مقصد_GPU(buft);

    واپس gpu->cached_بفر_type.max_size;
}

ساکن size_t ggml_backend_remoting_بفر_type_get_alloc_سائز(ggml_backend_بفر_type_t buft,
                                                               ثابت ggml_tensor *        tensor) {
    virtgpu * gpu = BUFT_مقصد_GPU(buft);

    اگر (tensor->بفر == تہی || !tensor->بفر->context ||
        !buft->device->iface.supports_buft(buft->device, tensor->بفر->buft)) {
        واپس ggml_nbytes(tensor);
    }

    واپس apir_بفر_type_get_alloc_سائز(gpu, gpu->cached_بفر_type.host_handle, tensor);
}

ثابت ggml_backend_بفر_type_i ggml_backend_remoting_بفر_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_بفر_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_remoting_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ تہی,
    /* .get_alignment       = */ ggml_backend_remoting_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_بفر_type_get_max_سائز,
    /* .get_alloc_size      = */ ggml_backend_remoting_بفر_type_get_alloc_سائز,
    /* .get_alloc_size_n    = */ تہی,
    /* .is_host             = */ تہی,
};

ثابت ggml_backend_بفر_type_i ggml_backend_remoting_بفر_from_ptr_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_بفر_type_get_name,
    /* .alloc_buffer        = */ تہی,
    /* .alloc_buffer_n      = */ تہی,
    /* .get_alignment       = */ ggml_backend_remoting_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_بفر_type_get_max_سائز,
    /* .get_alloc_size      = */ ggml_backend_remoting_بفر_type_get_alloc_سائز,
    /* .get_alloc_size_n    = */ تہی,
    /* .is_host             = */ تہی,
};
