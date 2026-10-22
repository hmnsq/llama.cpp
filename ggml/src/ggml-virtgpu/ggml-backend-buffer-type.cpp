#شامل "ggml-remoting.h"

ایستا ggml_backend_buffer_t ggml_backend_remoting_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                            size_t                     size) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) malloc(اندازه(*context));
    اگر (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu = gpu;

    منطقی async__unused, host_buffer__unused, events__unused, mmap_support__unused;
    منطقی buffer_from_host_ptr;
    apir_device_get_props(gpu, &async__unused, &host_buffer__unused, &buffer_from_host_ptr, &events__unused, &mmap_support__unused);

    اگر (buffer_from_host_ptr) {
        context->apir_context = apir_device_buffer_from_ptr(gpu, size, size);
        context->base         = context->apir_context.shmem.mmap_ptr;
        context->is_from_ptr  = درست;
    } وگرنه {
        context->apir_context = apir_buffer_type_alloc_buffer(gpu, gpu->cached_buffer_type.host_handle, size);
        context->is_from_ptr  = نادرست;
        context->base         = پوچ;
    }

    ggml_backend_buffer_t buffer =
        ggml_backend_buffer_init(buft, ggml_backend_remoting_buffer_interface, (تهی *) context, size);

    بازگشت buffer;
}

ایستا ثابت نویسه * ggml_backend_remoting_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    // Return the prefixed name that was built once during initialization
    بازگشت gpu->cached_buffer_type.name;
}

ایستا size_t ggml_backend_remoting_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    بازگشت gpu->cached_buffer_type.alignment;
}

ایستا size_t ggml_backend_remoting_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    بازگشت gpu->cached_buffer_type.max_size;
}

ایستا size_t ggml_backend_remoting_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                               ثابت ggml_tensor *        tensor) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    اگر (tensor->buffer == پوچ || !tensor->buffer->context ||
        !buft->device->iface.supports_buft(buft->device, tensor->buffer->buft)) {
        بازگشت ggml_nbytes(tensor);
    }

    بازگشت apir_buffer_type_get_alloc_size(gpu, gpu->cached_buffer_type.host_handle, tensor);
}

ثابت ggml_backend_buffer_type_i ggml_backend_remoting_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_remoting_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ پوچ,
    /* .get_alignment       = */ ggml_backend_remoting_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_remoting_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ پوچ,
    /* .is_host             = */ پوچ,
};

ثابت ggml_backend_buffer_type_i ggml_backend_remoting_buffer_from_ptr_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_buffer_type_get_name,
    /* .alloc_buffer        = */ پوچ,
    /* .alloc_buffer_n      = */ پوچ,
    /* .get_alignment       = */ ggml_backend_remoting_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_remoting_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ پوچ,
    /* .is_host             = */ پوچ,
};
