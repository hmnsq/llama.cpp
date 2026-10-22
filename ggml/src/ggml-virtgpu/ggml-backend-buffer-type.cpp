#ଅନ୍ତର୍ଭାଵ "ggml-remoting.h"

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_remoting_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                            size_t                     size) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) malloc(ଆକାର(*context));
    ଯଦି (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu = gpu;

    ତାର୍କିକ async__unused, host_buffer__unused, events__unused, mmap_support__unused;
    ତାର୍କିକ buffer_from_host_ptr;
    apir_device_get_props(gpu, &async__unused, &host_buffer__unused, &buffer_from_host_ptr, &events__unused, &mmap_support__unused);

    ଯଦି (buffer_from_host_ptr) {
        context->apir_context = apir_device_buffer_from_ptr(gpu, size, size);
        context->base         = context->apir_context.shmem.mmap_ptr;
        context->is_from_ptr  = ସତ୍ୟ;
    } ଅନ୍ୟଥା {
        context->apir_context = apir_buffer_type_alloc_buffer(gpu, gpu->cached_buffer_type.host_handle, size);
        context->is_from_ptr  = ମିଥ୍ୟା;
        context->base         = ଖାଲି;
    }

    ggml_backend_buffer_t buffer =
        ggml_backend_buffer_init(buft, ggml_backend_remoting_buffer_interface, (ଶୂନ୍ୟ *) context, size);

    ଫେରନ୍ତୁ buffer;
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_remoting_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    // Return the prefixed name that was built once during initialization
    ଫେରନ୍ତୁ gpu->cached_buffer_type.name;
}

ସ୍ଥିର size_t ggml_backend_remoting_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    ଫେରନ୍ତୁ gpu->cached_buffer_type.alignment;
}

ସ୍ଥିର size_t ggml_backend_remoting_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    ଫେରନ୍ତୁ gpu->cached_buffer_type.max_size;
}

ସ୍ଥିର size_t ggml_backend_remoting_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                               ଅଚଳ ggml_tensor *        tensor) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    ଯଦି (tensor->buffer == ଖାଲି || !tensor->buffer->context ||
        !buft->device->iface.supports_buft(buft->device, tensor->buffer->buft)) {
        ଫେରନ୍ତୁ ggml_nbytes(tensor);
    }

    ଫେରନ୍ତୁ apir_buffer_type_get_alloc_size(gpu, gpu->cached_buffer_type.host_handle, tensor);
}

ଅଚଳ ggml_backend_buffer_type_i ggml_backend_remoting_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_remoting_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ଖାଲି,
    /* .get_alignment       = */ ggml_backend_remoting_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_remoting_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ଖାଲି,
    /* .is_host             = */ ଖାଲି,
};

ଅଚଳ ggml_backend_buffer_type_i ggml_backend_remoting_buffer_from_ptr_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_buffer_type_get_name,
    /* .alloc_buffer        = */ ଖାଲି,
    /* .alloc_buffer_n      = */ ଖାଲି,
    /* .get_alignment       = */ ggml_backend_remoting_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_remoting_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ଖାଲି,
    /* .is_host             = */ ଖାଲି,
};
