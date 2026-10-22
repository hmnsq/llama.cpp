#অন্তর্ভাৱ "ggml-remoting.h"

#অন্তর্ভাৱ <mutex>

স্থির ধ্রুৱ ৱর্ণ * ggml_backend_remoting_device_get_name(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    // Return the prefixed name that was built once during initialization
    ঘূৰাই gpu->cached_device_info.name;
}

স্থির ধ্রুৱ ৱর্ণ * ggml_backend_remoting_device_get_description(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    // Return the pre-cached description from the virtgpu structure
    ঘূৰাই gpu->cached_device_info.description;
}

স্থির গণনা ggml_backend_dev_type ggml_backend_remoting_device_get_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    ঘূৰাই (গণনা ggml_backend_dev_type) gpu->cached_device_info.type;
}

স্থির শূন্য ggml_backend_remoting_device_get_memory(ggml_backend_dev_t dev, size_t * মুক্ত, size_t * total) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    *মুক্ত  = gpu->cached_device_info.memory_free;
    *total = gpu->cached_device_info.memory_total;
}

স্থির তার্কিক ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, ধ্রুৱ ggml_tensor * op) {
#যদি USE_ALWAYS_TRUE_SUPPORTS_OP == 1
    /* ggml-rpc cheats it like this */
    /* with the current implementation of serialize_tensor, the src/view aren't properly passed */
    UNUSED(dev);
    UNUSED(op);

    ঘূৰাই সঁচা;
#অন্যথা
    virtgpu * gpu = DEV_TO_GPU(dev);

    ঘূৰাই apir_device_supports_op(gpu, op);
#endif
}

স্থির তার্কিক ggml_backend_remoting_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    তার্কিক supported = buft->device == dev;

    ঘূৰাই supported;
}

স্থির তার্কিক ggml_backend_remoting_device_offload_op(ggml_backend_dev_t dev, ধ্রুৱ ggml_tensor * op) {
    UNUSED(dev);
    UNUSED(op);

    ঘূৰাই মিছা;
}

স্থির শূন্য ggml_backend_remoting_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_remoting_device_get_name(dev);
    props->description = ggml_backend_remoting_device_get_description(dev);
    props->type        = ggml_backend_remoting_device_get_type(dev);
    ggml_backend_remoting_device_get_memory(dev, &props->memory_free, &props->memory_total);

    virtgpu * gpu = DEV_TO_GPU(dev);
    apir_device_get_props(gpu, &props->caps.async, &props->caps.host_buffer, &props->caps.buffer_from_host_ptr,
                          &props->caps.events, &props->caps.mmap_support);

    props->caps.buffer_from_host_ptr = মিছা;
    props->caps.async                = মিছা;
    props->caps.events               = মিছা;
}

ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    স্থির std::atomic<তার্কিক>        initialized = মিছা;
    স্থির ggml_backend_buffer_type buft;

    যদি (!initialized) {
        স্থির std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        যদি (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_buffer_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (শূন্য *) gpu->cached_buffer_type.host_handle,
            };
            initialized = সঁচা;
        }
    }

    ঘূৰাই &buft;
}

স্থির ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_from_ptr_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    স্থির std::atomic<তার্কিক>        initialized = মিছা;
    স্থির ggml_backend_buffer_type buft;

    যদি (!initialized) {
        স্থির std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        যদি (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_buffer_from_ptr_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (শূন্য *) gpu->cached_buffer_type.host_handle,
            };
            initialized = সঁচা;
        }
    }

    ঘূৰাই &buft;
}

স্থির ggml_backend_buffer_t ggml_backend_remoting_device_buffer_from_ptr(ggml_backend_dev_t dev,
                                                                          শূন্য *             ptr,
                                                                          size_t             size,
                                                                          size_t             max_tensor_size) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) malloc(পরিমাণ(*context));
    যদি (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu          = gpu;
    context->apir_context = apir_device_buffer_from_ptr(gpu, size, max_tensor_size);
    context->base         = ptr;
    context->is_from_ptr  = সঁচা;

    ggml_backend_buffer_t buffer =
        ggml_backend_buffer_init(ggml_backend_remoting_device_get_buffer_from_ptr_type(dev),
                                 ggml_backend_remoting_buffer_from_ptr_interface, (শূন্য *) context, size);

    ঘূৰাই buffer;
}

ধ্রুৱ ggml_backend_device_i ggml_backend_remoting_device_interface = {
    /* .get_name             = */ ggml_backend_remoting_device_get_name,
    /* .get_description      = */ ggml_backend_remoting_device_get_description,
    /* .get_memory           = */ ggml_backend_remoting_device_get_memory,
    /* .get_type             = */ ggml_backend_remoting_device_get_type,
    /* .get_props            = */ ggml_backend_remoting_device_get_props,
    /* .init_backend         = */ ggml_backend_remoting_device_init,
    /* .get_buffer_type      = */ ggml_backend_remoting_device_get_buffer_type,
    /* .get_host_buffer_type = */ রিক্ত,
    /* .buffer_from_host_ptr = */ ggml_backend_remoting_device_buffer_from_ptr,
    /* .supports_op          = */ ggml_backend_remoting_device_supports_op,
    /* .supports_buft        = */ ggml_backend_remoting_device_supports_buft,
    /* .offload_op           = */ ggml_backend_remoting_device_offload_op,
    /* .event_new            = */ রিক্ত,
    /* .event_free           = */ রিক্ত,
    /* .event_synchronize    = */ রিক্ত,
};
