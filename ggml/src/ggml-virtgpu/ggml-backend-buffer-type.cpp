#अन्तर्भाव "ggml-remoting.h"

स्थिर ggml_backend_बफर_t ggml_backend_remoting_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft,
                                                                            size_t                     size) {
    virtgpu * gpu = BUFT_गन्तव्यम्_GPU(buft);

    ggml_backend_remoting_बफर_context * context = (ggml_backend_remoting_बफर_context *) स्मृतिं_देहि(परिमाण(*context));
    यदि (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu = gpu;

    तार्किक async_अप्रयुक्तम्, host_बफर_अप्रयुक्तम्, events_अप्रयुक्तम्, mmap_support_अप्रयुक्तम्;
    तार्किक बफर_from_host_ptr;
    apir_device_get_props(gpu, &async_अप्रयुक्तम्, &host_बफर_अप्रयुक्तम्, &बफर_from_host_ptr, &events_अप्रयुक्तम्, &mmap_support_अप्रयुक्तम्);

    यदि (बफर_from_host_ptr) {
        context->apir_context = apir_device_बफर_from_ptr(gpu, size, size);
        context->आधारः         = context->apir_context.shmem.mmap_ptr;
        context->is_from_ptr  = सत्य;
    } अन्यथा {
        context->apir_context = apir_बफर_type_alloc_बफर(gpu, gpu->cached_बफर_type.host_handle, size);
        context->is_from_ptr  = असत्य;
        context->आधारः         = रिक्त;
    }

    ggml_backend_बफर_t बफर =
        ggml_backend_बफर_init(buft, ggml_backend_remoting_बफर_interface, (शून्य *) context, size);

    प्रत्यागम बफर;
}

स्थिर ध्रुव वर्ण * ggml_backend_remoting_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    virtgpu * gpu = BUFT_गन्तव्यम्_GPU(buft);

    // Return the prefixed name that was built once during initialization
    प्रत्यागम gpu->cached_बफर_type.name;
}

स्थिर size_t ggml_backend_remoting_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    virtgpu * gpu = BUFT_गन्तव्यम्_GPU(buft);

    प्रत्यागम gpu->cached_बफर_type.alignment;
}

स्थिर size_t ggml_backend_remoting_बफर_type_get_max_आकार(ggml_backend_बफर_type_t buft) {
    virtgpu * gpu = BUFT_गन्तव्यम्_GPU(buft);

    प्रत्यागम gpu->cached_बफर_type.max_size;
}

स्थिर size_t ggml_backend_remoting_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft,
                                                               ध्रुव ggml_tensor *        tensor) {
    virtgpu * gpu = BUFT_गन्तव्यम्_GPU(buft);

    यदि (tensor->बफर == रिक्त || !tensor->बफर->context ||
        !buft->device->iface.supports_buft(buft->device, tensor->बफर->buft)) {
        प्रत्यागम ggml_nbytes(tensor);
    }

    प्रत्यागम apir_बफर_type_get_alloc_आकार(gpu, gpu->cached_बफर_type.host_handle, tensor);
}

ध्रुव ggml_backend_बफर_type_i ggml_backend_remoting_बफर_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_बफर_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_remoting_बफर_type_alloc_बफर,
    /* .alloc_buffer_n      = */ रिक्त,
    /* .get_alignment       = */ ggml_backend_remoting_बफर_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_बफर_type_get_max_आकार,
    /* .get_alloc_size      = */ ggml_backend_remoting_बफर_type_get_alloc_आकार,
    /* .get_alloc_size_n    = */ रिक्त,
    /* .is_host             = */ रिक्त,
};

ध्रुव ggml_backend_बफर_type_i ggml_backend_remoting_बफर_from_ptr_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_बफर_type_get_name,
    /* .alloc_buffer        = */ रिक्त,
    /* .alloc_buffer_n      = */ रिक्त,
    /* .get_alignment       = */ ggml_backend_remoting_बफर_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_बफर_type_get_max_आकार,
    /* .get_alloc_size      = */ ggml_backend_remoting_बफर_type_get_alloc_आकार,
    /* .get_alloc_size_n    = */ रिक्त,
    /* .is_host             = */ रिक्त,
};
