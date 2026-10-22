#ضمن "ggml-remoting.h"

ساكن ggml_backend_بفر_t ggml_backend_remoting_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft,
                                                                            size_t                     size) {
    virtgpu * gpu = BUFT_منزل_GPU(buft);

    ggml_backend_remoting_بفر_context * context = (ggml_backend_remoting_بفر_context *) يادگيري_ڏيو(حجم(*context));
    إذا (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu = gpu;

    منطقي async_اڻ_استعمال_ٿيل, host_بفر_اڻ_استعمال_ٿيل, events_اڻ_استعمال_ٿيل, mmap_support_اڻ_استعمال_ٿيل;
    منطقي بفر_from_host_ptr;
    apir_device_get_props(gpu, &async_اڻ_استعمال_ٿيل, &host_بفر_اڻ_استعمال_ٿيل, &بفر_from_host_ptr, &events_اڻ_استعمال_ٿيل, &mmap_support_اڻ_استعمال_ٿيل);

    إذا (بفر_from_host_ptr) {
        context->apir_context = apir_device_بفر_from_ptr(gpu, size, size);
        context->بنياد         = context->apir_context.shmem.mmap_ptr;
        context->is_from_ptr  = سچ;
    } وإلا {
        context->apir_context = apir_بفر_type_alloc_بفر(gpu, gpu->cached_بفر_type.host_handle, size);
        context->is_from_ptr  = خطأ;
        context->بنياد         = عدم;
    }

    ggml_backend_بفر_t بفر =
        ggml_backend_بفر_init(buft, ggml_backend_remoting_بفر_interface, (فراغ *) context, size);

    أرجع بفر;
}

ساكن ثابت حرف * ggml_backend_remoting_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    virtgpu * gpu = BUFT_منزل_GPU(buft);

    // Return the prefixed name that was built once during initialization
    أرجع gpu->cached_بفر_type.name;
}

ساكن size_t ggml_backend_remoting_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    virtgpu * gpu = BUFT_منزل_GPU(buft);

    أرجع gpu->cached_بفر_type.alignment;
}

ساكن size_t ggml_backend_remoting_بفر_type_get_max_سائيز(ggml_backend_بفر_type_t buft) {
    virtgpu * gpu = BUFT_منزل_GPU(buft);

    أرجع gpu->cached_بفر_type.max_size;
}

ساكن size_t ggml_backend_remoting_بفر_type_get_alloc_سائيز(ggml_backend_بفر_type_t buft,
                                                               ثابت ggml_tensor *        tensor) {
    virtgpu * gpu = BUFT_منزل_GPU(buft);

    إذا (tensor->بفر == عدم || !tensor->بفر->context ||
        !buft->device->iface.supports_buft(buft->device, tensor->بفر->buft)) {
        أرجع ggml_nbytes(tensor);
    }

    أرجع apir_بفر_type_get_alloc_سائيز(gpu, gpu->cached_بفر_type.host_handle, tensor);
}

ثابت ggml_backend_بفر_type_i ggml_backend_remoting_بفر_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_بفر_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_remoting_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ عدم,
    /* .get_alignment       = */ ggml_backend_remoting_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_بفر_type_get_max_سائيز,
    /* .get_alloc_size      = */ ggml_backend_remoting_بفر_type_get_alloc_سائيز,
    /* .get_alloc_size_n    = */ عدم,
    /* .is_host             = */ عدم,
};

ثابت ggml_backend_بفر_type_i ggml_backend_remoting_بفر_from_ptr_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_بفر_type_get_name,
    /* .alloc_buffer        = */ عدم,
    /* .alloc_buffer_n      = */ عدم,
    /* .get_alignment       = */ ggml_backend_remoting_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_بفر_type_get_max_سائيز,
    /* .get_alloc_size      = */ ggml_backend_remoting_بفر_type_get_alloc_سائيز,
    /* .get_alloc_size_n    = */ عدم,
    /* .is_host             = */ عدم,
};
