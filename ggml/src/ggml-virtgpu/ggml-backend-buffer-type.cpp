#ضمن "ggml-remoting.h"

ساكن ggml_backend_buffer_t ggml_backend_remoting_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                            size_t                     size) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) malloc(حجم(*context));
    ܐܢ (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu = gpu;

    منطقي async__unused, host_buffer__unused, events__unused, mmap_support__unused;
    منطقي buffer_from_host_ptr;
    apir_device_get_props(gpu, &async__unused, &host_buffer__unused, &buffer_from_host_ptr, &events__unused, &mmap_support__unused);

    ܐܢ (buffer_from_host_ptr) {
        context->apir_context = apir_device_buffer_from_ptr(gpu, size, size);
        context->base         = context->apir_context.shmem.mmap_ptr;
        context->is_from_ptr  = ܫܪܪܐ;
    } ܐܚܪܢܐ {
        context->apir_context = apir_buffer_type_alloc_buffer(gpu, gpu->cached_buffer_type.host_handle, size);
        context->is_from_ptr  = ܕܓܠܐ;
        context->base         = عدم;
    }

    ggml_backend_buffer_t buffer =
        ggml_backend_buffer_init(buft, ggml_backend_remoting_buffer_interface, (فراغ *) context, size);

    ܦܢܐ buffer;
}

ساكن ثابت حرف * ggml_backend_remoting_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    // Return the prefixed name that was built once during initialization
    ܦܢܐ gpu->cached_buffer_type.name;
}

ساكن size_t ggml_backend_remoting_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    ܦܢܐ gpu->cached_buffer_type.alignment;
}

ساكن size_t ggml_backend_remoting_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    ܦܢܐ gpu->cached_buffer_type.max_size;
}

ساكن size_t ggml_backend_remoting_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                               ثابت ggml_tensor *        tensor) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    ܐܢ (tensor->buffer == عدم || !tensor->buffer->context ||
        !buft->device->iface.supports_buft(buft->device, tensor->buffer->buft)) {
        ܦܢܐ ggml_nbytes(tensor);
    }

    ܦܢܐ apir_buffer_type_get_alloc_size(gpu, gpu->cached_buffer_type.host_handle, tensor);
}

ثابت ggml_backend_buffer_type_i ggml_backend_remoting_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_remoting_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ عدم,
    /* .get_alignment       = */ ggml_backend_remoting_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_remoting_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ عدم,
    /* .is_host             = */ عدم,
};

ثابت ggml_backend_buffer_type_i ggml_backend_remoting_buffer_from_ptr_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_buffer_type_get_name,
    /* .alloc_buffer        = */ عدم,
    /* .alloc_buffer_n      = */ عدم,
    /* .get_alignment       = */ ggml_backend_remoting_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_remoting_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ عدم,
    /* .is_host             = */ عدم,
};
