#ಅನ್ತರ್ಭಾವ "ggml-remoting.h"

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_remoting_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                            size_t                     size) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) malloc(ಪರಿಮಾಣ(*context));
    ಯದಿ (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu = gpu;

    ತಾರ್ಕಿಕ async__unused, host_buffer__unused, events__unused, mmap_support__unused;
    ತಾರ್ಕಿಕ buffer_from_host_ptr;
    apir_device_get_props(gpu, &async__unused, &host_buffer__unused, &buffer_from_host_ptr, &events__unused, &mmap_support__unused);

    ಯದಿ (buffer_from_host_ptr) {
        context->apir_context = apir_device_buffer_from_ptr(gpu, size, size);
        context->base         = context->apir_context.shmem.mmap_ptr;
        context->is_from_ptr  = ಸತ್ಯ;
    } ಅನ್ಯಥಾ {
        context->apir_context = apir_buffer_type_alloc_buffer(gpu, gpu->cached_buffer_type.host_handle, size);
        context->is_from_ptr  = ಅಸತ್ಯ;
        context->base         = ರಿಕ್ತ;
    }

    ggml_backend_buffer_t buffer =
        ggml_backend_buffer_init(buft, ggml_backend_remoting_buffer_interface, (ಶೂನ್ಯ *) context, size);

    ಪ್ರತ್ಯಾಗಮ buffer;
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_remoting_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    // Return the prefixed name that was built once during initialization
    ಪ್ರತ್ಯಾಗಮ gpu->cached_buffer_type.name;
}

ಸ್ಥಿರ size_t ggml_backend_remoting_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    ಪ್ರತ್ಯಾಗಮ gpu->cached_buffer_type.alignment;
}

ಸ್ಥಿರ size_t ggml_backend_remoting_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    ಪ್ರತ್ಯಾಗಮ gpu->cached_buffer_type.max_size;
}

ಸ್ಥಿರ size_t ggml_backend_remoting_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                               ಧ್ರುವ ggml_tensor *        tensor) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    ಯದಿ (tensor->buffer == ರಿಕ್ತ || !tensor->buffer->context ||
        !buft->device->iface.supports_buft(buft->device, tensor->buffer->buft)) {
        ಪ್ರತ್ಯಾಗಮ ggml_nbytes(tensor);
    }

    ಪ್ರತ್ಯಾಗಮ apir_buffer_type_get_alloc_size(gpu, gpu->cached_buffer_type.host_handle, tensor);
}

ಧ್ರುವ ggml_backend_buffer_type_i ggml_backend_remoting_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_remoting_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ರಿಕ್ತ,
    /* .get_alignment       = */ ggml_backend_remoting_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_remoting_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ರಿಕ್ತ,
    /* .is_host             = */ ರಿಕ್ತ,
};

ಧ್ರುವ ggml_backend_buffer_type_i ggml_backend_remoting_buffer_from_ptr_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_buffer_type_get_name,
    /* .alloc_buffer        = */ ರಿಕ್ತ,
    /* .alloc_buffer_n      = */ ರಿಕ್ತ,
    /* .get_alignment       = */ ggml_backend_remoting_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_remoting_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ರಿಕ್ತ,
    /* .is_host             = */ ರಿಕ್ತ,
};
