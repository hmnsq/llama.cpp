#अन्तर्भाव "ggml-remoting.h"

स्थिर ggml_backend_buffer_t ggml_backend_remoting_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                            माप_प्रकार                     size) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) दो_स्मृति(आकार(*context));
    जर (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu = gpu;

    तार्किक async__unused, host_buffer__unused, events__unused, mmap_support__unused;
    तार्किक buffer_from_host_ptr;
    apir_device_get_props(gpu, &async__unused, &host_buffer__unused, &buffer_from_host_ptr, &events__unused, &mmap_support__unused);

    जर (buffer_from_host_ptr) {
        context->apir_context = apir_device_buffer_from_ptr(gpu, size, size);
        context->base         = context->apir_context.shmem.mmap_ptr;
        context->is_from_ptr  = सत्य;
    } अन्यथा {
        context->apir_context = apir_buffer_type_alloc_buffer(gpu, gpu->cached_buffer_type.host_handle, size);
        context->is_from_ptr  = असत्य;
        context->base         = रिक्त;
    }

    ggml_backend_buffer_t buffer =
        ggml_backend_buffer_init(buft, ggml_backend_remoting_buffer_interface, (शून्य *) context, size);

    परत buffer;
}

स्थिर स्थिरांक अक्षर * ggml_backend_remoting_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    // Return the prefixed name that was built once during initialization
    परत gpu->cached_buffer_type.name;
}

स्थिर माप_प्रकार ggml_backend_remoting_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    परत gpu->cached_buffer_type.alignment;
}

स्थिर माप_प्रकार ggml_backend_remoting_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    परत gpu->cached_buffer_type.max_size;
}

स्थिर माप_प्रकार ggml_backend_remoting_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                               स्थिरांक ggml_tensor *        tensor) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    जर (tensor->buffer == रिक्त || !tensor->buffer->context ||
        !buft->device->iface.supports_buft(buft->device, tensor->buffer->buft)) {
        परत ggml_nbytes(tensor);
    }

    परत apir_buffer_type_get_alloc_size(gpu, gpu->cached_buffer_type.host_handle, tensor);
}

स्थिरांक ggml_backend_buffer_type_i ggml_backend_remoting_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_remoting_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ रिक्त,
    /* .get_alignment       = */ ggml_backend_remoting_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_remoting_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ रिक्त,
    /* .is_host             = */ रिक्त,
};

स्थिरांक ggml_backend_buffer_type_i ggml_backend_remoting_buffer_from_ptr_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_buffer_type_get_name,
    /* .alloc_buffer        = */ रिक्त,
    /* .alloc_buffer_n      = */ रिक्त,
    /* .get_alignment       = */ ggml_backend_remoting_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_remoting_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ रिक्त,
    /* .is_host             = */ रिक्त,
};
