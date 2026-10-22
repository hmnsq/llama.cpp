#ಅನ್ತರ್ಭಾವ "ggml-remoting.h"

#ಅನ್ತರ್ಭಾವ <mutex>

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_remoting_device_get_name(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    // Return the prefixed name that was built once during initialization
    ಪ್ರತ್ಯಾಗಮ gpu->cached_device_info.name;
}

ಸ್ಥಿರ ಧ್ರುವ ವರ್ಣ * ggml_backend_remoting_device_get_description(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    // Return the pre-cached description from the virtgpu structure
    ಪ್ರತ್ಯಾಗಮ gpu->cached_device_info.description;
}

ಸ್ಥಿರ ಗಣನಾ ggml_backend_dev_type ggml_backend_remoting_device_get_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    ಪ್ರತ್ಯಾಗಮ (ಗಣನಾ ggml_backend_dev_type) gpu->cached_device_info.type;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_remoting_device_get_memory(ggml_backend_dev_t dev, size_t * ಮುಕ್ತ, size_t * total) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    *ಮುಕ್ತ  = gpu->cached_device_info.memory_free;
    *total = gpu->cached_device_info.memory_total;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, ಧ್ರುವ ggml_tensor * op) {
#ಯದಿ USE_ALWAYS_TRUE_SUPPORTS_OP == 1
    /* ggml-rpc cheats it like this */
    /* with the current implementation of serialize_tensor, the src/view aren't properly passed */
    UNUSED(dev);
    UNUSED(op);

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
#ಅನ್ಯಥಾ
    virtgpu * gpu = DEV_TO_GPU(dev);

    ಪ್ರತ್ಯಾಗಮ apir_device_supports_op(gpu, op);
#endif
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_remoting_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ತಾರ್ಕಿಕ supported = buft->device == dev;

    ಪ್ರತ್ಯಾಗಮ supported;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_remoting_device_offload_op(ggml_backend_dev_t dev, ಧ್ರುವ ggml_tensor * op) {
    UNUSED(dev);
    UNUSED(op);

    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_remoting_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_remoting_device_get_name(dev);
    props->description = ggml_backend_remoting_device_get_description(dev);
    props->type        = ggml_backend_remoting_device_get_type(dev);
    ggml_backend_remoting_device_get_memory(dev, &props->memory_free, &props->memory_total);

    virtgpu * gpu = DEV_TO_GPU(dev);
    apir_device_get_props(gpu, &props->caps.async, &props->caps.host_buffer, &props->caps.buffer_from_host_ptr,
                          &props->caps.events, &props->caps.mmap_support);

    props->caps.buffer_from_host_ptr = ಅಸತ್ಯ;
    props->caps.async                = ಅಸತ್ಯ;
    props->caps.events               = ಅಸತ್ಯ;
}

ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    ಸ್ಥಿರ std::atomic<ತಾರ್ಕಿಕ>        initialized = ಅಸತ್ಯ;
    ಸ್ಥಿರ ggml_backend_buffer_type buft;

    ಯದಿ (!initialized) {
        ಸ್ಥಿರ std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        ಯದಿ (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_buffer_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (ಶೂನ್ಯ *) gpu->cached_buffer_type.host_handle,
            };
            initialized = ಸತ್ಯ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ &buft;
}

ಸ್ಥಿರ ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_from_ptr_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    ಸ್ಥಿರ std::atomic<ತಾರ್ಕಿಕ>        initialized = ಅಸತ್ಯ;
    ಸ್ಥಿರ ggml_backend_buffer_type buft;

    ಯದಿ (!initialized) {
        ಸ್ಥಿರ std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        ಯದಿ (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_buffer_from_ptr_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (ಶೂನ್ಯ *) gpu->cached_buffer_type.host_handle,
            };
            initialized = ಸತ್ಯ;
        }
    }

    ಪ್ರತ್ಯಾಗಮ &buft;
}

ಸ್ಥಿರ ggml_backend_buffer_t ggml_backend_remoting_device_buffer_from_ptr(ggml_backend_dev_t dev,
                                                                          ಶೂನ್ಯ *             ptr,
                                                                          size_t             size,
                                                                          size_t             max_tensor_size) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) malloc(ಪರಿಮಾಣ(*context));
    ಯದಿ (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu          = gpu;
    context->apir_context = apir_device_buffer_from_ptr(gpu, size, max_tensor_size);
    context->base         = ptr;
    context->is_from_ptr  = ಸತ್ಯ;

    ggml_backend_buffer_t buffer =
        ggml_backend_buffer_init(ggml_backend_remoting_device_get_buffer_from_ptr_type(dev),
                                 ggml_backend_remoting_buffer_from_ptr_interface, (ಶೂನ್ಯ *) context, size);

    ಪ್ರತ್ಯಾಗಮ buffer;
}

ಧ್ರುವ ggml_backend_device_i ggml_backend_remoting_device_interface = {
    /* .get_name             = */ ggml_backend_remoting_device_get_name,
    /* .get_description      = */ ggml_backend_remoting_device_get_description,
    /* .get_memory           = */ ggml_backend_remoting_device_get_memory,
    /* .get_type             = */ ggml_backend_remoting_device_get_type,
    /* .get_props            = */ ggml_backend_remoting_device_get_props,
    /* .init_backend         = */ ggml_backend_remoting_device_init,
    /* .get_buffer_type      = */ ggml_backend_remoting_device_get_buffer_type,
    /* .get_host_buffer_type = */ ರಿಕ್ತ,
    /* .buffer_from_host_ptr = */ ggml_backend_remoting_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_remoting_device_supports_op,
    /* .supports_buft        = */ ggml_backend_remoting_device_supports_buft,
    /* .offload_op           = */ ggml_backend_remoting_device_offload_op,
    /* .event_new            = */ ರಿಕ್ತ,
    /* .event_free           = */ ರಿಕ್ತ,
    /* .event_synchronize    = */ ರಿಕ್ತ,
};
