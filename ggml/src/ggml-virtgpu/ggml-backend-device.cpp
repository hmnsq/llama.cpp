#അന്തര്ഭാവ "ggml-remoting.h"

#അന്തര്ഭാവ <mutex>

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_remoting_device_get_name(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    // Return the prefixed name that was built once during initialization
    തിരികെ gpu->cached_device_info.name;
}

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_remoting_device_get_description(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    // Return the pre-cached description from the virtgpu structure
    തിരികെ gpu->cached_device_info.description;
}

സ്ഥിതം എണ്ണൽ ggml_backend_dev_type ggml_backend_remoting_device_get_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    തിരികെ (എണ്ണൽ ggml_backend_dev_type) gpu->cached_device_info.type;
}

സ്ഥിതം ശൂന്യം ggml_backend_remoting_device_get_memory(ggml_backend_dev_t dev, size_t * സ്വതന്ത്രം, size_t * total) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    *സ്വതന്ത്രം  = gpu->cached_device_info.memory_free;
    *total = gpu->cached_device_info.memory_total;
}

സ്ഥിതം തർക്കം ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, സ്ഥിരം ggml_tensor * op) {
#എങ്കിൽ USE_ALWAYS_TRUE_SUPPORTS_OP == 1
    /* ggml-rpc cheats it like this */
    /* with the current implementation of serialize_tensor, the src/view aren't properly passed */
    UNUSED(dev);
    UNUSED(op);

    തിരികെ സത്യം;
#അല്ലെങ്കിൽ
    virtgpu * gpu = DEV_TO_GPU(dev);

    തിരികെ apir_device_supports_op(gpu, op);
#endif
}

സ്ഥിതം തർക്കം ggml_backend_remoting_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    തർക്കം supported = buft->device == dev;

    തിരികെ supported;
}

സ്ഥിതം തർക്കം ggml_backend_remoting_device_offload_op(ggml_backend_dev_t dev, സ്ഥിരം ggml_tensor * op) {
    UNUSED(dev);
    UNUSED(op);

    തിരികെ അസത്യം;
}

സ്ഥിതം ശൂന്യം ggml_backend_remoting_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_remoting_device_get_name(dev);
    props->description = ggml_backend_remoting_device_get_description(dev);
    props->type        = ggml_backend_remoting_device_get_type(dev);
    ggml_backend_remoting_device_get_memory(dev, &props->memory_free, &props->memory_total);

    virtgpu * gpu = DEV_TO_GPU(dev);
    apir_device_get_props(gpu, &props->caps.async, &props->caps.host_buffer, &props->caps.buffer_from_host_ptr,
                          &props->caps.events, &props->caps.mmap_support);

    props->caps.buffer_from_host_ptr = അസത്യം;
    props->caps.async                = അസത്യം;
    props->caps.events               = അസത്യം;
}

ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    സ്ഥിതം std::atomic<തർക്കം>        initialized = അസത്യം;
    സ്ഥിതം ggml_backend_buffer_type buft;

    എങ്കിൽ (!initialized) {
        സ്ഥിതം std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        എങ്കിൽ (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_buffer_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (ശൂന്യം *) gpu->cached_buffer_type.host_handle,
            };
            initialized = സത്യം;
        }
    }

    തിരികെ &buft;
}

സ്ഥിതം ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_from_ptr_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    സ്ഥിതം std::atomic<തർക്കം>        initialized = അസത്യം;
    സ്ഥിതം ggml_backend_buffer_type buft;

    എങ്കിൽ (!initialized) {
        സ്ഥിതം std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        എങ്കിൽ (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_buffer_from_ptr_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (ശൂന്യം *) gpu->cached_buffer_type.host_handle,
            };
            initialized = സത്യം;
        }
    }

    തിരികെ &buft;
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_remoting_device_buffer_from_ptr(ggml_backend_dev_t dev,
                                                                          ശൂന്യം *             ptr,
                                                                          size_t             size,
                                                                          size_t             max_tensor_size) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) malloc(വലുപ്പം(*context));
    എങ്കിൽ (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu          = gpu;
    context->apir_context = apir_device_buffer_from_ptr(gpu, size, max_tensor_size);
    context->base         = ptr;
    context->is_from_ptr  = സത്യം;

    ggml_backend_buffer_t buffer =
        ggml_backend_buffer_init(ggml_backend_remoting_device_get_buffer_from_ptr_type(dev),
                                 ggml_backend_remoting_buffer_from_ptr_interface, (ശൂന്യം *) context, size);

    തിരികെ buffer;
}

സ്ഥിരം ggml_backend_device_i ggml_backend_remoting_device_interface = {
    /* .get_name             = */ ggml_backend_remoting_device_get_name,
    /* .get_description      = */ ggml_backend_remoting_device_get_description,
    /* .get_memory           = */ ggml_backend_remoting_device_get_memory,
    /* .get_type             = */ ggml_backend_remoting_device_get_type,
    /* .get_props            = */ ggml_backend_remoting_device_get_props,
    /* .init_backend         = */ ggml_backend_remoting_device_init,
    /* .get_buffer_type      = */ ggml_backend_remoting_device_get_buffer_type,
    /* .get_host_buffer_type = */ ശൂന്യ,
    /* .buffer_from_host_ptr = */ ggml_backend_remoting_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_remoting_device_supports_op,
    /* .supports_buft        = */ ggml_backend_remoting_device_supports_buft,
    /* .offload_op           = */ ggml_backend_remoting_device_offload_op,
    /* .event_new            = */ ശൂന്യ,
    /* .event_free           = */ ശൂന്യ,
    /* .event_synchronize    = */ ശൂന്യ,
};
