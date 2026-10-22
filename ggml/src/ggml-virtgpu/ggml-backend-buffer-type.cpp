#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-remoting.h"

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_remoting_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft,
                                                                            size_t                     size) {
    virtgpu * gpu = BUFT_ਮੰਜ਼ਿਲ_GPU(buft);

    ggml_backend_remoting_ਬਫਰ_context * context = (ggml_backend_remoting_ਬਫਰ_context *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਆਕਾਰ(*context));
    ਜੇ (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu = gpu;

    ਤਾਰਕਿਕ async_ਅਣਵਰਤਿਆ, host_ਬਫਰ_ਅਣਵਰਤਿਆ, events_ਅਣਵਰਤਿਆ, mmap_support_ਅਣਵਰਤਿਆ;
    ਤਾਰਕਿਕ ਬਫਰ_from_host_ptr;
    apir_device_get_props(gpu, &async_ਅਣਵਰਤਿਆ, &host_ਬਫਰ_ਅਣਵਰਤਿਆ, &ਬਫਰ_from_host_ptr, &events_ਅਣਵਰਤਿਆ, &mmap_support_ਅਣਵਰਤਿਆ);

    ਜੇ (ਬਫਰ_from_host_ptr) {
        context->apir_context = apir_device_ਬਫਰ_from_ptr(gpu, size, size);
        context->ਅਧਾਰ         = context->apir_context.shmem.mmap_ptr;
        context->is_from_ptr  = ਸੱਚ;
    } ਨਹੀਂਤਾਂ {
        context->apir_context = apir_ਬਫਰ_type_alloc_ਬਫਰ(gpu, gpu->cached_ਬਫਰ_type.host_handle, size);
        context->is_from_ptr  = ਝੂਠ;
        context->ਅਧਾਰ         = ਨਿਰਾ;
    }

    ggml_backend_ਬਫਰ_t ਬਫਰ =
        ggml_backend_ਬਫਰ_init(buft, ggml_backend_remoting_ਬਫਰ_interface, (ਖਾਲੀ *) context, size);

    ਵਾਪਸ ਬਫਰ;
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_remoting_ਬਫਰ_type_get_name(ggml_backend_ਬਫਰ_type_t buft) {
    virtgpu * gpu = BUFT_ਮੰਜ਼ਿਲ_GPU(buft);

    // Return the prefixed name that was built once during initialization
    ਵਾਪਸ gpu->cached_ਬਫਰ_type.name;
}

ਸਥਿਰ size_t ggml_backend_remoting_ਬਫਰ_type_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    virtgpu * gpu = BUFT_ਮੰਜ਼ਿਲ_GPU(buft);

    ਵਾਪਸ gpu->cached_ਬਫਰ_type.alignment;
}

ਸਥਿਰ size_t ggml_backend_remoting_ਬਫਰ_type_get_max_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft) {
    virtgpu * gpu = BUFT_ਮੰਜ਼ਿਲ_GPU(buft);

    ਵਾਪਸ gpu->cached_ਬਫਰ_type.max_size;
}

ਸਥਿਰ size_t ggml_backend_remoting_ਬਫਰ_type_get_alloc_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft,
                                                               ਅਚਲ ggml_tensor *        tensor) {
    virtgpu * gpu = BUFT_ਮੰਜ਼ਿਲ_GPU(buft);

    ਜੇ (tensor->ਬਫਰ == ਨਿਰਾ || !tensor->ਬਫਰ->context ||
        !buft->device->iface.supports_buft(buft->device, tensor->ਬਫਰ->buft)) {
        ਵਾਪਸ ggml_nbytes(tensor);
    }

    ਵਾਪਸ apir_ਬਫਰ_type_get_alloc_ਮਾਪ(gpu, gpu->cached_ਬਫਰ_type.host_handle, tensor);
}

ਅਚਲ ggml_backend_ਬਫਰ_type_i ggml_backend_remoting_ਬਫਰ_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_ਬਫਰ_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_remoting_ਬਫਰ_type_alloc_ਬਫਰ,
    /* .alloc_buffer_n      = */ ਨਿਰਾ,
    /* .get_alignment       = */ ggml_backend_remoting_ਬਫਰ_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_ਬਫਰ_type_get_max_ਮਾਪ,
    /* .get_alloc_size      = */ ggml_backend_remoting_ਬਫਰ_type_get_alloc_ਮਾਪ,
    /* .get_alloc_size_n    = */ ਨਿਰਾ,
    /* .is_host             = */ ਨਿਰਾ,
};

ਅਚਲ ggml_backend_ਬਫਰ_type_i ggml_backend_remoting_ਬਫਰ_from_ptr_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_ਬਫਰ_type_get_name,
    /* .alloc_buffer        = */ ਨਿਰਾ,
    /* .alloc_buffer_n      = */ ਨਿਰਾ,
    /* .get_alignment       = */ ggml_backend_remoting_ਬਫਰ_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_ਬਫਰ_type_get_max_ਮਾਪ,
    /* .get_alloc_size      = */ ggml_backend_remoting_ਬਫਰ_type_get_alloc_ਮਾਪ,
    /* .get_alloc_size_n    = */ ਨਿਰਾ,
    /* .is_host             = */ ਨਿਰਾ,
};
