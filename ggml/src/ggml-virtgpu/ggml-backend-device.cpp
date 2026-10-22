#અન્તર્ભાવ "ggml-remoting.h"

#અન્તર્ભાવ <mutex>

સ્થિર અચલ અક્ષર * ggml_backend_remoting_device_get_name(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    // Return the prefixed name that was built once during initialization
    પાછા gpu->cached_device_info.name;
}

સ્થિર અચલ અક્ષર * ggml_backend_remoting_device_get_description(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    // Return the pre-cached description from the virtgpu structure
    પાછા gpu->cached_device_info.description;
}

સ્થિર ગણના ggml_backend_dev_type ggml_backend_remoting_device_get_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    પાછા (ગણના ggml_backend_dev_type) gpu->cached_device_info.type;
}

સ્થિર શૂન્ય ggml_backend_remoting_device_get_memory(ggml_backend_dev_t dev, size_t * મુક્ત, size_t * total) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    *મુક્ત  = gpu->cached_device_info.memory_free;
    *total = gpu->cached_device_info.memory_total;
}

સ્થિર તાર્કિક ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, અચલ ggml_tensor * op) {
#જો USE_ALWAYS_TRUE_SUPPORTS_OP == 1
    /* ggml-rpc cheats it like this */
    /* with the current implementation of serialize_tensor, the src/view aren't properly passed */
    UNUSED(dev);
    UNUSED(op);

    પાછા સાચું;
#નહીંતર
    virtgpu * gpu = DEV_TO_GPU(dev);

    પાછા apir_device_supports_op(gpu, op);
#endif
}

સ્થિર તાર્કિક ggml_backend_remoting_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    તાર્કિક supported = buft->device == dev;

    પાછા supported;
}

સ્થિર તાર્કિક ggml_backend_remoting_device_offload_op(ggml_backend_dev_t dev, અચલ ggml_tensor * op) {
    UNUSED(dev);
    UNUSED(op);

    પાછા ખોટું;
}

સ્થિર શૂન્ય ggml_backend_remoting_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_remoting_device_get_name(dev);
    props->description = ggml_backend_remoting_device_get_description(dev);
    props->type        = ggml_backend_remoting_device_get_type(dev);
    ggml_backend_remoting_device_get_memory(dev, &props->memory_free, &props->memory_total);

    virtgpu * gpu = DEV_TO_GPU(dev);
    apir_device_get_props(gpu, &props->caps.async, &props->caps.host_buffer, &props->caps.buffer_from_host_ptr,
                          &props->caps.events, &props->caps.mmap_support);

    props->caps.buffer_from_host_ptr = ખોટું;
    props->caps.async                = ખોટું;
    props->caps.events               = ખોટું;
}

ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    સ્થિર std::atomic<તાર્કિક>        initialized = ખોટું;
    સ્થિર ggml_backend_buffer_type buft;

    જો (!initialized) {
        સ્થિર std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        જો (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_buffer_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (શૂન્ય *) gpu->cached_buffer_type.host_handle,
            };
            initialized = સાચું;
        }
    }

    પાછા &buft;
}

સ્થિર ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_from_ptr_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    સ્થિર std::atomic<તાર્કિક>        initialized = ખોટું;
    સ્થિર ggml_backend_buffer_type buft;

    જો (!initialized) {
        સ્થિર std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        જો (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_buffer_from_ptr_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (શૂન્ય *) gpu->cached_buffer_type.host_handle,
            };
            initialized = સાચું;
        }
    }

    પાછા &buft;
}

સ્થિર ggml_backend_buffer_t ggml_backend_remoting_device_buffer_from_ptr(ggml_backend_dev_t dev,
                                                                          શૂન્ય *             ptr,
                                                                          size_t             size,
                                                                          size_t             max_tensor_size) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) malloc(કદ(*context));
    જો (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu          = gpu;
    context->apir_context = apir_device_buffer_from_ptr(gpu, size, max_tensor_size);
    context->base         = ptr;
    context->is_from_ptr  = સાચું;

    ggml_backend_buffer_t buffer =
        ggml_backend_buffer_init(ggml_backend_remoting_device_get_buffer_from_ptr_type(dev),
                                 ggml_backend_remoting_buffer_from_ptr_interface, (શૂન્ય *) context, size);

    પાછા buffer;
}

અચલ ggml_backend_device_i ggml_backend_remoting_device_interface = {
    /* .get_name             = */ ggml_backend_remoting_device_get_name,
    /* .get_description      = */ ggml_backend_remoting_device_get_description,
    /* .get_memory           = */ ggml_backend_remoting_device_get_memory,
    /* .get_type             = */ ggml_backend_remoting_device_get_type,
    /* .get_props            = */ ggml_backend_remoting_device_get_props,
    /* .init_backend         = */ ggml_backend_remoting_device_init,
    /* .get_buffer_type      = */ ggml_backend_remoting_device_get_buffer_type,
    /* .get_host_buffer_type = */ ખાલી,
    /* .buffer_from_host_ptr = */ ggml_backend_remoting_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_remoting_device_supports_op,
    /* .supports_buft        = */ ggml_backend_remoting_device_supports_buft,
    /* .offload_op           = */ ggml_backend_remoting_device_offload_op,
    /* .event_new            = */ ખાલી,
    /* .event_free           = */ ખાલી,
    /* .event_synchronize    = */ ખાલી,
};
