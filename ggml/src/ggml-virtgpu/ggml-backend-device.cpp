#অন্তর্ভাব "ggml-remoting.h"

#অন্তর্ভাব <mutex>

স্থির ধ্রুবক অক্ষর * ggml_backend_remoting_device_get_name(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    // Return the prefixed name that was built once during initialization
    ফেরত gpu->cached_device_info.name;
}

স্থির ধ্রুবক অক্ষর * ggml_backend_remoting_device_get_description(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    // Return the pre-cached description from the virtgpu structure
    ফেরত gpu->cached_device_info.description;
}

স্থির গণনা ggml_backend_dev_type ggml_backend_remoting_device_get_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    ফেরত (গণনা ggml_backend_dev_type) gpu->cached_device_info.type;
}

স্থির শূন্য ggml_backend_remoting_device_get_memory(ggml_backend_dev_t dev, size_t * মুক্ত, size_t * যোগ) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    *মুক্ত  = gpu->cached_device_info.memory_free;
    *যোগ = gpu->cached_device_info.memory_যোগ;
}

স্থির যৌক্তিক ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, ধ্রুবক ggml_tensor * op) {
#যদি USE_ALWAYS_TRUE_SUPPORTS_OP == 1
    /* ggml-rpc cheats it like this */
    /* with the current implementation of serialize_tensor, the src/view aren't properly passed */
    UNUSED(dev);
    UNUSED(op);

    ফেরত সত্য;
#অন্যথা
    virtgpu * gpu = DEV_TO_GPU(dev);

    ফেরত apir_device_supports_op(gpu, op);
#endif
}

স্থির যৌক্তিক ggml_backend_remoting_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_বাফার_type_t buft) {
    যৌক্তিক supported = buft->device == dev;

    ফেরত supported;
}

স্থির যৌক্তিক ggml_backend_remoting_device_offload_op(ggml_backend_dev_t dev, ধ্রুবক ggml_tensor * op) {
    UNUSED(dev);
    UNUSED(op);

    ফেরত মিথ্যা;
}

স্থির শূন্য ggml_backend_remoting_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_remoting_device_get_name(dev);
    props->description = ggml_backend_remoting_device_get_description(dev);
    props->type        = ggml_backend_remoting_device_get_type(dev);
    ggml_backend_remoting_device_get_memory(dev, &props->memory_free, &props->memory_যোগ);

    virtgpu * gpu = DEV_TO_GPU(dev);
    apir_device_get_props(gpu, &props->caps.async, &props->caps.host_বাফার, &props->caps.বাফার_from_host_ptr,
                          &props->caps.events, &props->caps.mmap_support);

    props->caps.বাফার_from_host_ptr = মিথ্যা;
    props->caps.async                = মিথ্যা;
    props->caps.events               = মিথ্যা;
}

ggml_backend_বাফার_type_t ggml_backend_remoting_device_get_বাফার_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    স্থির std::atomic<যৌক্তিক>        initialized = মিথ্যা;
    স্থির ggml_backend_বাফার_type buft;

    যদি (!initialized) {
        স্থির std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        যদি (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_বাফার_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (শূন্য *) gpu->cached_বাফার_type.host_handle,
            };
            initialized = সত্য;
        }
    }

    ফেরত &buft;
}

স্থির ggml_backend_বাফার_type_t ggml_backend_remoting_device_get_বাফার_from_ptr_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    স্থির std::atomic<যৌক্তিক>        initialized = মিথ্যা;
    স্থির ggml_backend_বাফার_type buft;

    যদি (!initialized) {
        স্থির std::mutex           mutex;
        std::lock_guard<std::mutex> lock(mutex);

        যদি (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_বাফার_from_ptr_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (শূন্য *) gpu->cached_বাফার_type.host_handle,
            };
            initialized = সত্য;
        }
    }

    ফেরত &buft;
}

স্থির ggml_backend_বাফার_t ggml_backend_remoting_device_বাফার_from_ptr(ggml_backend_dev_t dev,
                                                                          শূন্য *             ptr,
                                                                          size_t             size,
                                                                          size_t             max_tensor_আকার) {
    virtgpu * gpu = DEV_TO_GPU(dev);

    ggml_backend_remoting_বাফার_context * context = (ggml_backend_remoting_বাফার_context *) malloc(আকার(*context));
    যদি (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu          = gpu;
    context->apir_context = apir_device_বাফার_from_ptr(gpu, size, max_tensor_আকার);
    context->base         = ptr;
    context->is_from_ptr  = সত্য;

    ggml_backend_বাফার_t বাফার =
        ggml_backend_বাফার_init(ggml_backend_remoting_device_get_বাফার_from_ptr_type(dev),
                                 ggml_backend_remoting_বাফার_from_ptr_interface, (শূন্য *) context, size);

    ফেরত বাফার;
}

ধ্রুবক ggml_backend_device_i ggml_backend_remoting_device_interface = {
    /* .get_name             = */ ggml_backend_remoting_device_get_name,
    /* .get_description      = */ ggml_backend_remoting_device_get_description,
    /* .get_memory           = */ ggml_backend_remoting_device_get_memory,
    /* .get_type             = */ ggml_backend_remoting_device_get_type,
    /* .get_props            = */ ggml_backend_remoting_device_get_props,
    /* .init_backend         = */ ggml_backend_remoting_device_init,
    /* .get_buffer_type      = */ ggml_backend_remoting_device_get_বাফার_type,
    /* .get_host_buffer_type = */ ফাঁকা,
    /* .buffer_from_host_ptr = */ ggml_backend_remoting_device_বাফার_from_ptr,
    /* .supports_op          = */ ggml_backend_remoting_device_supports_op,
    /* .supports_buft        = */ ggml_backend_remoting_device_supports_buft,
    /* .offload_op           = */ ggml_backend_remoting_device_offload_op,
    /* .event_new            = */ ফাঁকা,
    /* .event_free           = */ ফাঁকা,
    /* .event_synchronize    = */ ফাঁকা,
};
