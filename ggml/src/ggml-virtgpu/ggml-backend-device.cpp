#अन्तर्भाव "ggml-remoting.h"

#अन्तर्भाव <mutex>

स्थिर ध्रुव वर्ण * ggml_backend_remoting_device_get_name(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    // Return the prefixed name that was built once during initialization
    पटिनिवत्त gpu->cached_device_info.name;
}

स्थिर ध्रुव वर्ण * ggml_backend_remoting_device_get_description(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    // Return the pre-cached description from the virtgpu structure
    पटिनिवत्त gpu->cached_device_info.description;
}

स्थिर गणना ggml_backend_dev_type ggml_backend_remoting_device_get_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    पटिनिवत्त (गणना ggml_backend_dev_type) gpu->cached_device_info.type;
}

स्थिर शून्य ggml_backend_remoting_device_get_memory(ggml_backend_dev_t dev, size_t * मुक्त, size_t * total) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    *मुक्त  = gpu->cached_device_info.memory_free;
    *total = gpu->cached_device_info.memory_total;
}

स्थिर तार्किक ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, ध्रुव ggml_tensor * op) {
#सचे USE_ALWAYS_TRUE_SUPPORTS_OP == 1
    /* ggml-rpc cheats it like this */
    /* with the current implementation of serialize_tensor, the src/view aren't properly passed */
    UNUSED(dev);
    UNUSED(op);

    पटिनिवत्त सच्च;
#अञ्ञथा
    virtgpu * gpu = DEV_TO_GPU(dev);

    पटिनिवत्त apir_device_supports_op(gpu, op);
#endif
}

स्थिर तार्किक ggml_backend_remoting_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    तार्किक supported = buft->device == dev;

    पटिनिवत्त supported;
}

स्थिर तार्किक ggml_backend_remoting_device_offload_op(ggml_backend_dev_t dev, ध्रुव ggml_tensor * op) {
    UNUSED(dev);
    UNUSED(op);

    पटिनिवत्त मिच्छा;
}

स्थिर शून्य ggml_backend_remoting_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_remoting_device_get_name(dev);
    props->description = ggml_backend_remoting_device_get_description(dev);
    props->type        = ggml_backend_remoting_device_get_type(dev);
    ggml_backend_remoting_device_get_memory(dev, &props->memory_free, &props->memory_total);

    virtgpu * gpu = DEV_TO_GPU(dev);
    apir_device_get_props(gpu, &props->caps.async, &props->caps.host_buffer, &props->caps.buffer_from_host_ptr,
                          &props->caps.events, &props->caps.mmap_support);

    props->caps.buffer_from_host_ptr = मिच्छा;
    props->caps.async                = मिच्छा;
    props->caps.events               = मिच्छा;
}

ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    स्थिर std::atomic<तार्किक>        initialized = मिच्छा;
    स्थिर ggml_backend_buffer_type buft;

    सचे (!initialized) {
        स्थिर std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        सचे (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_buffer_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (शून्य *) gpu->cached_buffer_type.host_handle,
            };
            initialized = सच्च;
        }
    }

    पटिनिवत्त &buft;
}

स्थिर ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_from_ptr_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    स्थिर std::atomic<तार्किक>        initialized = मिच्छा;
    स्थिर ggml_backend_buffer_type buft;

    सचे (!initialized) {
        स्थिर std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        सचे (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_buffer_from_ptr_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (शून्य *) gpu->cached_buffer_type.host_handle,
            };
            initialized = सच्च;
        }
    }

    पटिनिवत्त &buft;
}

स्थिर ggml_backend_buffer_t ggml_backend_remoting_device_buffer_from_ptr(ggml_backend_dev_t dev,
                                                                          शून्य *             ptr,
                                                                          size_t             size,
                                                                          size_t             max_tensor_size) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) malloc(परिमाण(*context));
    सचे (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu          = gpu;
    context->apir_context = apir_device_buffer_from_ptr(gpu, size, max_tensor_size);
    context->base         = ptr;
    context->is_from_ptr  = सच्च;

    ggml_backend_buffer_t buffer =
        ggml_backend_buffer_init(ggml_backend_remoting_device_get_buffer_from_ptr_type(dev),
                                 ggml_backend_remoting_buffer_from_ptr_interface, (शून्य *) context, size);

    पटिनिवत्त buffer;
}

ध्रुव ggml_backend_device_i ggml_backend_remoting_device_interface = {
    /* .get_name             = */ ggml_backend_remoting_device_get_name,
    /* .get_description      = */ ggml_backend_remoting_device_get_description,
    /* .get_memory           = */ ggml_backend_remoting_device_get_memory,
    /* .get_type             = */ ggml_backend_remoting_device_get_type,
    /* .get_props            = */ ggml_backend_remoting_device_get_props,
    /* .init_backend         = */ ggml_backend_remoting_device_init,
    /* .get_buffer_type      = */ ggml_backend_remoting_device_get_buffer_type,
    /* .get_host_buffer_type = */ रिक्त,
    /* .buffer_from_host_ptr = */ ggml_backend_remoting_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_remoting_device_supports_op,
    /* .supports_buft        = */ ggml_backend_remoting_device_supports_buft,
    /* .offload_op           = */ ggml_backend_remoting_device_offload_op,
    /* .event_new            = */ रिक्त,
    /* .event_free           = */ रिक्त,
    /* .event_synchronize    = */ रिक्त,
};
