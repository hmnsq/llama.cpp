#அந்தர்பாவ "ggml-remoting.h"

#அந்தர்பாவ <mutex>

நிலையான மாறிலி எழுத்து * ggml_backend_remoting_device_get_name(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    // Return the prefixed name that was built once during initialization
    திரும்பு gpu->cached_device_info.name;
}

நிலையான மாறிலி எழுத்து * ggml_backend_remoting_device_get_description(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    // Return the pre-cached description from the virtgpu structure
    திரும்பு gpu->cached_device_info.description;
}

நிலையான எண்ணிக்கை ggml_backend_dev_type ggml_backend_remoting_device_get_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    திரும்பு (எண்ணிக்கை ggml_backend_dev_type) gpu->cached_device_info.type;
}

நிலையான வெறுமை ggml_backend_remoting_device_get_memory(ggml_backend_dev_t dev, size_t * விடுவி, size_t * total) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    *விடுவி  = gpu->cached_device_info.memory_free;
    *total = gpu->cached_device_info.memory_total;
}

நிலையான தருக்க ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, மாறிலி ggml_tensor * op) {
#எனில் USE_ALWAYS_TRUE_SUPPORTS_OP == 1
    /* ggml-rpc cheats it like this */
    /* with the current implementation of serialize_tensor, the src/view aren't properly passed */
    UNUSED(dev);
    UNUSED(op);

    திரும்பு உண்மை;
#இல்லையெனில்
    virtgpu * gpu = DEV_TO_GPU(dev);

    திரும்பு apir_device_supports_op(gpu, op);
#endif
}

நிலையான தருக்க ggml_backend_remoting_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    தருக்க supported = buft->device == dev;

    திரும்பு supported;
}

நிலையான தருக்க ggml_backend_remoting_device_offload_op(ggml_backend_dev_t dev, மாறிலி ggml_tensor * op) {
    UNUSED(dev);
    UNUSED(op);

    திரும்பு பொய்;
}

நிலையான வெறுமை ggml_backend_remoting_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_remoting_device_get_name(dev);
    props->description = ggml_backend_remoting_device_get_description(dev);
    props->type        = ggml_backend_remoting_device_get_type(dev);
    ggml_backend_remoting_device_get_memory(dev, &props->memory_free, &props->memory_total);

    virtgpu * gpu = DEV_TO_GPU(dev);
    apir_device_get_props(gpu, &props->caps.async, &props->caps.host_buffer, &props->caps.buffer_from_host_ptr,
                          &props->caps.events, &props->caps.mmap_support);

    props->caps.buffer_from_host_ptr = பொய்;
    props->caps.async                = பொய்;
    props->caps.events               = பொய்;
}

ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    நிலையான std::atomic<தருக்க>        initialized = பொய்;
    நிலையான ggml_backend_buffer_type buft;

    எனில் (!initialized) {
        நிலையான std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        எனில் (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_buffer_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (வெறுமை *) gpu->cached_buffer_type.host_handle,
            };
            initialized = உண்மை;
        }
    }

    திரும்பு &buft;
}

நிலையான ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_from_ptr_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    நிலையான std::atomic<தருக்க>        initialized = பொய்;
    நிலையான ggml_backend_buffer_type buft;

    எனில் (!initialized) {
        நிலையான std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        எனில் (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_buffer_from_ptr_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (வெறுமை *) gpu->cached_buffer_type.host_handle,
            };
            initialized = உண்மை;
        }
    }

    திரும்பு &buft;
}

நிலையான ggml_backend_buffer_t ggml_backend_remoting_device_buffer_from_ptr(ggml_backend_dev_t dev,
                                                                          வெறுமை *             ptr,
                                                                          size_t             size,
                                                                          size_t             max_tensor_size) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) malloc(அளவு(*context));
    எனில் (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu          = gpu;
    context->apir_context = apir_device_buffer_from_ptr(gpu, size, max_tensor_size);
    context->base         = ptr;
    context->is_from_ptr  = உண்மை;

    ggml_backend_buffer_t buffer =
        ggml_backend_buffer_init(ggml_backend_remoting_device_get_buffer_from_ptr_type(dev),
                                 ggml_backend_remoting_buffer_from_ptr_interface, (வெறுமை *) context, size);

    திரும்பு buffer;
}

மாறிலி ggml_backend_device_i ggml_backend_remoting_device_interface = {
    /* .get_name             = */ ggml_backend_remoting_device_get_name,
    /* .get_description      = */ ggml_backend_remoting_device_get_description,
    /* .get_memory           = */ ggml_backend_remoting_device_get_memory,
    /* .get_type             = */ ggml_backend_remoting_device_get_type,
    /* .get_props            = */ ggml_backend_remoting_device_get_props,
    /* .init_backend         = */ ggml_backend_remoting_device_init,
    /* .get_buffer_type      = */ ggml_backend_remoting_device_get_buffer_type,
    /* .get_host_buffer_type = */ வெற்று,
    /* .buffer_from_host_ptr = */ ggml_backend_remoting_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_remoting_device_supports_op,
    /* .supports_buft        = */ ggml_backend_remoting_device_supports_buft,
    /* .offload_op           = */ ggml_backend_remoting_device_offload_op,
    /* .event_new            = */ வெற்று,
    /* .event_free           = */ வெற்று,
    /* .event_synchronize    = */ வெற்று,
};
