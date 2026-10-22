#समावेश "ggml-remoting.h"

अटल ggml_backend_बफर_t ggml_backend_remoting_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft,
                                                                            माप_प्रकार                     size) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    ggml_backend_remoting_बफर_context * context = (ggml_backend_remoting_बफर_context *) दो_स्मृति(माप(*context));
    अगर (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu = gpu;

    bool async__unused, host_बफर_unused, events__unused, mmap_support__unused;
    bool बफर_from_host_ptr;
    apir_device_get_props(gpu, &async__unused, &host_बफर_unused, &बफर_from_host_ptr, &events__unused, &mmap_support__unused);

    अगर (बफर_from_host_ptr) {
        context->apir_context = apir_device_बफर_from_ptr(gpu, size, size);
        context->base         = context->apir_context.shmem.mmap_ptr;
        context->is_from_ptr  = true;
    } अन्यथा {
        context->apir_context = apir_बफर_type_alloc_बफर(gpu, gpu->cached_बफर_type.host_handle, size);
        context->is_from_ptr  = false;
        context->base         = शून्य;
    }

    ggml_backend_बफर_t बफर =
        ggml_backend_बफर_init(buft, ggml_backend_remoting_बफर_interface, (व्योम *) context, size);

    वापस बफर;
}

अटल स्थिर अक्षर * ggml_backend_remoting_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    // Return the prefixed name that was built once during initialization
    वापस gpu->cached_बफर_type.name;
}

अटल माप_प्रकार ggml_backend_remoting_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    वापस gpu->cached_बफर_type.alignment;
}

अटल माप_प्रकार ggml_backend_remoting_बफर_type_get_max_आकार(ggml_backend_बफर_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    वापस gpu->cached_बफर_type.max_size;
}

अटल माप_प्रकार ggml_backend_remoting_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft,
                                                               स्थिर ggml_tensor *        tensor) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    अगर (tensor->बफर == शून्य || !tensor->बफर->context ||
        !buft->device->iface.supports_buft(buft->device, tensor->बफर->buft)) {
        वापस ggml_nbytes(tensor);
    }

    वापस apir_बफर_type_get_alloc_आकार(gpu, gpu->cached_बफर_type.host_handle, tensor);
}

स्थिर ggml_backend_बफर_type_i ggml_backend_remoting_बफर_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_बफर_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_remoting_बफर_type_alloc_बफर,
    /* .alloc_buffer_n      = */ शून्य,
    /* .get_alignment       = */ ggml_backend_remoting_बफर_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_बफर_type_get_max_आकार,
    /* .get_alloc_size      = */ ggml_backend_remoting_बफर_type_get_alloc_आकार,
    /* .get_alloc_size_n    = */ शून्य,
    /* .is_host             = */ शून्य,
};

स्थिर ggml_backend_बफर_type_i ggml_backend_remoting_बफर_from_ptr_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_बफर_type_get_name,
    /* .alloc_buffer        = */ शून्य,
    /* .alloc_buffer_n      = */ शून्य,
    /* .get_alignment       = */ ggml_backend_remoting_बफर_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_बफर_type_get_max_आकार,
    /* .get_alloc_size      = */ ggml_backend_remoting_बफर_type_get_alloc_आकार,
    /* .get_alloc_size_n    = */ शून्य,
    /* .is_host             = */ शून्य,
};
