#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-remoting.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <mutex>

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_remoting_device_get_name(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_ਮੰਜ਼ਿਲ_GPU(dev);

    // Return the prefixed name that was built once during initialization
    ਵਾਪਸ gpu->cached_device_info.name;
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_remoting_device_get_description(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_ਮੰਜ਼ਿਲ_GPU(dev);

    // Return the pre-cached description from the virtgpu structure
    ਵਾਪਸ gpu->cached_device_info.description;
}

ਸਥਿਰ ਗਿਣਤੀ ggml_backend_dev_type ggml_backend_remoting_device_get_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_ਮੰਜ਼ਿਲ_GPU(dev);

    ਵਾਪਸ (ਗਿਣਤੀ ggml_backend_dev_type) gpu->cached_device_info.type;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_remoting_device_get_ਸਮ੍ਰਿਤੀ(ggml_backend_dev_t dev, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
    virtgpu * gpu = DEV_ਮੰਜ਼ਿਲ_GPU(dev);

    *ਮੁਕਤ  = gpu->cached_device_info.ਸਮ੍ਰਿਤੀ_free;
    *ਕੁੱਲ = gpu->cached_device_info.ਸਮ੍ਰਿਤੀ_ਕੁੱਲ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, ਅਚਲ ggml_tensor * op) {
#ਜੇ USE_ALWAYS_TRUE_SUPPORTS_OP == 1
    /* ggml-rpc cheats it like this */
    /* with the current implementation of serialize_tensor, the src/view aren't properly passed */
    ਅਣਵਰਤਿਆ(dev);
    ਅਣਵਰਤਿਆ(op);

    ਵਾਪਸ ਸੱਚ;
#ਨਹੀਂਤਾਂ
    virtgpu * gpu = DEV_ਮੰਜ਼ਿਲ_GPU(dev);

    ਵਾਪਸ apir_device_supports_op(gpu, op);
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_remoting_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ਬਫਰ_type_t buft) {
    ਤਾਰਕਿਕ supported = buft->device == dev;

    ਵਾਪਸ supported;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_remoting_device_offload_op(ggml_backend_dev_t dev, ਅਚਲ ggml_tensor * op) {
    ਅਣਵਰਤਿਆ(dev);
    ਅਣਵਰਤਿਆ(op);

    ਵਾਪਸ ਝੂਠ;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_remoting_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_remoting_device_get_name(dev);
    props->description = ggml_backend_remoting_device_get_description(dev);
    props->type        = ggml_backend_remoting_device_get_type(dev);
    ggml_backend_remoting_device_get_ਸਮ੍ਰਿਤੀ(dev, &props->ਸਮ੍ਰਿਤੀ_free, &props->ਸਮ੍ਰਿਤੀ_ਕੁੱਲ);

    virtgpu * gpu = DEV_ਮੰਜ਼ਿਲ_GPU(dev);
    apir_device_get_props(gpu, &props->caps.async, &props->caps.host_ਬਫਰ, &props->caps.ਬਫਰ_from_host_ptr,
                          &props->caps.events, &props->caps.mmap_support);

    props->caps.ਬਫਰ_from_host_ptr = ਝੂਠ;
    props->caps.async                = ਝੂਠ;
    props->caps.events               = ਝੂਠ;
}

ggml_backend_ਬਫਰ_type_t ggml_backend_remoting_device_get_ਬਫਰ_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_ਮੰਜ਼ਿਲ_GPU(dev);

    ਸਥਿਰ std::atomic<ਤਾਰਕਿਕ>        initialized = ਝੂਠ;
    ਸਥਿਰ ggml_backend_ਬਫਰ_type buft;

    ਜੇ (!initialized) {
        ਸਥਿਰ std::mutex           mutex;
        std::lock_guard<std::mutex> ਤਾਲਾ(mutex);

        ਜੇ (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_ਬਫਰ_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (ਖਾਲੀ *) gpu->cached_ਬਫਰ_type.host_handle,
            };
            initialized = ਸੱਚ;
        }
    }

    ਵਾਪਸ &buft;
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_remoting_device_get_ਬਫਰ_from_ptr_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_ਮੰਜ਼ਿਲ_GPU(dev);

    ਸਥਿਰ std::atomic<ਤਾਰਕਿਕ>        initialized = ਝੂਠ;
    ਸਥਿਰ ggml_backend_ਬਫਰ_type buft;

    ਜੇ (!initialized) {
        ਸਥਿਰ std::mutex           mutex;
        std::lock_guard<std::mutex> ਤਾਲਾ(mutex);

        ਜੇ (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_ਬਫਰ_from_ptr_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (ਖਾਲੀ *) gpu->cached_ਬਫਰ_type.host_handle,
            };
            initialized = ਸੱਚ;
        }
    }

    ਵਾਪਸ &buft;
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_remoting_device_ਬਫਰ_from_ptr(ggml_backend_dev_t dev,
                                                                          ਖਾਲੀ *             ptr,
                                                                          size_t             size,
                                                                          size_t             max_tensor_ਮਾਪ) {
    virtgpu * gpu = DEV_ਮੰਜ਼ਿਲ_GPU(dev);

    ggml_backend_remoting_ਬਫਰ_context * context = (ggml_backend_remoting_ਬਫਰ_context *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਆਕਾਰ(*context));
    ਜੇ (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu          = gpu;
    context->apir_context = apir_device_ਬਫਰ_from_ptr(gpu, size, max_tensor_ਮਾਪ);
    context->ਅਧਾਰ         = ptr;
    context->is_from_ptr  = ਸੱਚ;

    ggml_backend_ਬਫਰ_t ਬਫਰ =
        ggml_backend_ਬਫਰ_init(ggml_backend_remoting_device_get_ਬਫਰ_from_ptr_type(dev),
                                 ggml_backend_remoting_ਬਫਰ_from_ptr_interface, (ਖਾਲੀ *) context, size);

    ਵਾਪਸ ਬਫਰ;
}

ਅਚਲ ggml_backend_device_i ggml_backend_remoting_device_interface = {
    /* .get_name             = */ ggml_backend_remoting_device_get_name,
    /* .get_description      = */ ggml_backend_remoting_device_get_description,
    /* .get_memory           = */ ggml_backend_remoting_device_get_ਸਮ੍ਰਿਤੀ,
    /* .get_type             = */ ggml_backend_remoting_device_get_type,
    /* .get_props            = */ ggml_backend_remoting_device_get_props,
    /* .init_backend         = */ ggml_backend_remoting_device_init,
    /* .get_buffer_type      = */ ggml_backend_remoting_device_get_ਬਫਰ_type,
    /* .get_host_buffer_type = */ ਨਿਰਾ,
    /* .buffer_from_host_ptr = */ ggml_backend_remoting_device_ਬਫਰ_from_ptr,
    /* .supports_op          = */ ggml_backend_remoting_device_supports_op,
    /* .supports_buft        = */ ggml_backend_remoting_device_supports_buft,
    /* .offload_op           = */ ggml_backend_remoting_device_offload_op,
    /* .event_new            = */ ਨਿਰਾ,
    /* .event_free           = */ ਨਿਰਾ,
    /* .event_synchronize    = */ ਨਿਰਾ,
};
