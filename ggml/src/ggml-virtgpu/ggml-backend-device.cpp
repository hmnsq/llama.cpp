#ଅନ୍ତର୍ଭାଵ "ggml-remoting.h"

#ଅନ୍ତର୍ଭାଵ <mutex>

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_remoting_device_get_name(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    // Return the prefixed name that was built once during initialization
    ଫେରନ୍ତୁ gpu->cached_device_info.name;
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_remoting_device_get_description(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    // Return the pre-cached description from the virtgpu structure
    ଫେରନ୍ତୁ gpu->cached_device_info.description;
}

ସ୍ଥିର ଗଣନା ggml_backend_dev_type ggml_backend_remoting_device_get_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    ଫେରନ୍ତୁ (ଗଣନା ggml_backend_dev_type) gpu->cached_device_info.type;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_remoting_device_get_memory(ggml_backend_dev_t dev, size_t * ମୁକ୍ତ, size_t * total) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    *ମୁକ୍ତ  = gpu->cached_device_info.memory_free;
    *total = gpu->cached_device_info.memory_total;
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, ଅଚଳ ggml_tensor * op) {
#ଯଦି USE_ALWAYS_TRUE_SUPPORTS_OP == 1
    /* ggml-rpc cheats it like this */
    /* with the current implementation of serialize_tensor, the src/view aren't properly passed */
    UNUSED(dev);
    UNUSED(op);

    ଫେରନ୍ତୁ ସତ୍ୟ;
#ଅନ୍ୟଥା
    virtgpu * gpu = DEV_TO_GPU(dev);

    ଫେରନ୍ତୁ apir_device_supports_op(gpu, op);
#endif
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_remoting_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ତାର୍କିକ supported = buft->device == dev;

    ଫେରନ୍ତୁ supported;
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_remoting_device_offload_op(ggml_backend_dev_t dev, ଅଚଳ ggml_tensor * op) {
    UNUSED(dev);
    UNUSED(op);

    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_remoting_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_remoting_device_get_name(dev);
    props->description = ggml_backend_remoting_device_get_description(dev);
    props->type        = ggml_backend_remoting_device_get_type(dev);
    ggml_backend_remoting_device_get_memory(dev, &props->memory_free, &props->memory_total);

    virtgpu * gpu = DEV_TO_GPU(dev);
    apir_device_get_props(gpu, &props->caps.async, &props->caps.host_buffer, &props->caps.buffer_from_host_ptr,
                          &props->caps.events, &props->caps.mmap_support);

    props->caps.buffer_from_host_ptr = ମିଥ୍ୟା;
    props->caps.async                = ମିଥ୍ୟା;
    props->caps.events               = ମିଥ୍ୟା;
}

ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    ସ୍ଥିର std::atomic<ତାର୍କିକ>        initialized = ମିଥ୍ୟା;
    ସ୍ଥିର ggml_backend_buffer_type buft;

    ଯଦି (!initialized) {
        ସ୍ଥିର std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        ଯଦି (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_buffer_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (ଶୂନ୍ୟ *) gpu->cached_buffer_type.host_handle,
            };
            initialized = ସତ୍ୟ;
        }
    }

    ଫେରନ୍ତୁ &buft;
}

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_from_ptr_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    ସ୍ଥିର std::atomic<ତାର୍କିକ>        initialized = ମିଥ୍ୟା;
    ସ୍ଥିର ggml_backend_buffer_type buft;

    ଯଦି (!initialized) {
        ସ୍ଥିର std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        ଯଦି (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_buffer_from_ptr_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (ଶୂନ୍ୟ *) gpu->cached_buffer_type.host_handle,
            };
            initialized = ସତ୍ୟ;
        }
    }

    ଫେରନ୍ତୁ &buft;
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_remoting_device_buffer_from_ptr(ggml_backend_dev_t dev,
                                                                          ଶୂନ୍ୟ *             ptr,
                                                                          size_t             size,
                                                                          size_t             max_tensor_size) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) malloc(ଆକାର(*context));
    ଯଦି (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu          = gpu;
    context->apir_context = apir_device_buffer_from_ptr(gpu, size, max_tensor_size);
    context->base         = ptr;
    context->is_from_ptr  = ସତ୍ୟ;

    ggml_backend_buffer_t buffer =
        ggml_backend_buffer_init(ggml_backend_remoting_device_get_buffer_from_ptr_type(dev),
                                 ggml_backend_remoting_buffer_from_ptr_interface, (ଶୂନ୍ୟ *) context, size);

    ଫେରନ୍ତୁ buffer;
}

ଅଚଳ ggml_backend_device_i ggml_backend_remoting_device_interface = {
    /* .get_name             = */ ggml_backend_remoting_device_get_name,
    /* .get_description      = */ ggml_backend_remoting_device_get_description,
    /* .get_memory           = */ ggml_backend_remoting_device_get_memory,
    /* .get_type             = */ ggml_backend_remoting_device_get_type,
    /* .get_props            = */ ggml_backend_remoting_device_get_props,
    /* .init_backend         = */ ggml_backend_remoting_device_init,
    /* .get_buffer_type      = */ ggml_backend_remoting_device_get_buffer_type,
    /* .get_host_buffer_type = */ ଖାଲି,
    /* .buffer_from_host_ptr = */ ggml_backend_remoting_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_remoting_device_supports_op,
    /* .supports_buft        = */ ggml_backend_remoting_device_supports_buft,
    /* .offload_op           = */ ggml_backend_remoting_device_offload_op,
    /* .event_new            = */ ଖାଲି,
    /* .event_free           = */ ଖାଲି,
    /* .event_synchronize    = */ ଖାଲି,
};
