#समावेश "ggml-remoting.h"

#समावेश <mutex>

अटल स्थिर अक्षर * ggml_backend_remoting_device_get_name(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    // Return the prefixed name that was built once during initialization
    वापस gpu->cached_device_info.name;
}

अटल स्थिर अक्षर * ggml_backend_remoting_device_get_description(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    // Return the pre-cached description from the virtgpu structure
    वापस gpu->cached_device_info.description;
}

अटल क्रमागत ggml_backend_dev_type ggml_backend_remoting_device_get_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    वापस (क्रमागत ggml_backend_dev_type) gpu->cached_device_info.type;
}

अटल व्योम ggml_backend_remoting_device_get_memory(ggml_backend_dev_t dev, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    *मुक्त  = gpu->cached_device_info.memory_free;
    *योग = gpu->cached_device_info.memory_योग;
}

अटल bool ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, स्थिर ggml_tensor * op) {
#अगर USE_ALWAYS_TRUE_SUPPORTS_OP == 1
    /* ggml-rpc cheats it like this */
    /* with the current implementation of serialize_tensor, the src/view aren't properly passed */
    UNUSED(dev);
    UNUSED(op);

    वापस true;
#अन्यथा
    virtgpu * gpu = DEV_TO_GPU(dev);

    वापस apir_device_supports_op(gpu, op);
#पूर्ण_अगर
}

अटल bool ggml_backend_remoting_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    bool supported = buft->device == dev;

    वापस supported;
}

अटल bool ggml_backend_remoting_device_offload_op(ggml_backend_dev_t dev, स्थिर ggml_tensor * op) {
    UNUSED(dev);
    UNUSED(op);

    वापस false;
}

अटल व्योम ggml_backend_remoting_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_remoting_device_get_name(dev);
    props->description = ggml_backend_remoting_device_get_description(dev);
    props->type        = ggml_backend_remoting_device_get_type(dev);
    ggml_backend_remoting_device_get_memory(dev, &props->memory_free, &props->memory_योग);

    virtgpu * gpu = DEV_TO_GPU(dev);
    apir_device_get_props(gpu, &props->caps.async, &props->caps.host_बफर, &props->caps.बफर_from_host_ptr,
                          &props->caps.events, &props->caps.mmap_support);

    props->caps.बफर_from_host_ptr = false;
    props->caps.async                = false;
    props->caps.events               = false;
}

ggml_backend_बफर_type_t ggml_backend_remoting_device_get_बफर_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    अटल std::atomic<bool>        initialized = false;
    अटल ggml_backend_बफर_type buft;

    अगर (!initialized) {
        अटल std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        अगर (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_बफर_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (व्योम *) gpu->cached_बफर_type.host_handle,
            };
            initialized = true;
        }
    }

    वापस &buft;
}

अटल ggml_backend_बफर_type_t ggml_backend_remoting_device_get_बफर_from_ptr_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    अटल std::atomic<bool>        initialized = false;
    अटल ggml_backend_बफर_type buft;

    अगर (!initialized) {
        अटल std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        अगर (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_बफर_from_ptr_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (व्योम *) gpu->cached_बफर_type.host_handle,
            };
            initialized = true;
        }
    }

    वापस &buft;
}

अटल ggml_backend_बफर_t ggml_backend_remoting_device_बफर_from_ptr(ggml_backend_dev_t dev,
                                                                          व्योम *             ptr,
                                                                          माप_प्रकार             size,
                                                                          माप_प्रकार             max_tensor_आकार) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    ggml_backend_remoting_बफर_context * context = (ggml_backend_remoting_बफर_context *) दो_स्मृति(माप(*context));
    अगर (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu          = gpu;
    context->apir_context = apir_device_बफर_from_ptr(gpu, size, max_tensor_आकार);
    context->base         = ptr;
    context->is_from_ptr  = true;

    ggml_backend_बफर_t बफर =
        ggml_backend_बफर_init(ggml_backend_remoting_device_get_बफर_from_ptr_type(dev),
                                 ggml_backend_remoting_बफर_from_ptr_interface, (व्योम *) context, size);

    वापस बफर;
}

स्थिर ggml_backend_device_i ggml_backend_remoting_device_interface = {
    /* .get_name             = */ ggml_backend_remoting_device_get_name,
    /* .get_description      = */ ggml_backend_remoting_device_get_description,
    /* .get_memory           = */ ggml_backend_remoting_device_get_memory,
    /* .get_type             = */ ggml_backend_remoting_device_get_type,
    /* .get_props            = */ ggml_backend_remoting_device_get_props,
    /* .init_backend         = */ ggml_backend_remoting_device_init,
    /* .get_buffer_type      = */ ggml_backend_remoting_device_get_बफर_type,
    /* .get_host_buffer_type = */ शून्य,
    /* .buffer_from_host_ptr = */ ggml_backend_remoting_device_बफर_from_ptr,
    /* .supports_op          = */ ggml_backend_remoting_device_supports_op,
    /* .supports_buft        = */ ggml_backend_remoting_device_supports_buft,
    /* .offload_op           = */ ggml_backend_remoting_device_offload_op,
    /* .event_new            = */ शून्य,
    /* .event_free           = */ शून्य,
    /* .event_synchronize    = */ शून्य,
};
