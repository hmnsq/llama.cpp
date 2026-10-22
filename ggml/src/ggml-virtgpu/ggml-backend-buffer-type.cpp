#அந்தர்பாவ "ggml-remoting.h"

நிலையான ggml_backend_buffer_t ggml_backend_remoting_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                            size_t                     size) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) malloc(அளவு(*context));
    எனில் (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu = gpu;

    தருக்க async__unused, host_buffer__unused, events__unused, mmap_support__unused;
    தருக்க buffer_from_host_ptr;
    apir_device_get_props(gpu, &async__unused, &host_buffer__unused, &buffer_from_host_ptr, &events__unused, &mmap_support__unused);

    எனில் (buffer_from_host_ptr) {
        context->apir_context = apir_device_buffer_from_ptr(gpu, size, size);
        context->base         = context->apir_context.shmem.mmap_ptr;
        context->is_from_ptr  = உண்மை;
    } இல்லையெனில் {
        context->apir_context = apir_buffer_type_alloc_buffer(gpu, gpu->cached_buffer_type.host_handle, size);
        context->is_from_ptr  = பொய்;
        context->base         = வெற்று;
    }

    ggml_backend_buffer_t buffer =
        ggml_backend_buffer_init(buft, ggml_backend_remoting_buffer_interface, (வெறுமை *) context, size);

    திரும்பு buffer;
}

நிலையான மாறிலி எழுத்து * ggml_backend_remoting_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    // Return the prefixed name that was built once during initialization
    திரும்பு gpu->cached_buffer_type.name;
}

நிலையான size_t ggml_backend_remoting_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    திரும்பு gpu->cached_buffer_type.alignment;
}

நிலையான size_t ggml_backend_remoting_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    திரும்பு gpu->cached_buffer_type.max_size;
}

நிலையான size_t ggml_backend_remoting_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                               மாறிலி ggml_tensor *        tensor) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    எனில் (tensor->buffer == வெற்று || !tensor->buffer->context ||
        !buft->device->iface.supports_buft(buft->device, tensor->buffer->buft)) {
        திரும்பு ggml_nbytes(tensor);
    }

    திரும்பு apir_buffer_type_get_alloc_size(gpu, gpu->cached_buffer_type.host_handle, tensor);
}

மாறிலி ggml_backend_buffer_type_i ggml_backend_remoting_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_remoting_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ வெற்று,
    /* .get_alignment       = */ ggml_backend_remoting_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_remoting_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ வெற்று,
    /* .is_host             = */ வெற்று,
};

மாறிலி ggml_backend_buffer_type_i ggml_backend_remoting_buffer_from_ptr_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_buffer_type_get_name,
    /* .alloc_buffer        = */ வெற்று,
    /* .alloc_buffer_n      = */ வெற்று,
    /* .get_alignment       = */ ggml_backend_remoting_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_remoting_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ வெற்று,
    /* .is_host             = */ வெற்று,
};
