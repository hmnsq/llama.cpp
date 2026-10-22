#ضمن "ggml-remoting.h"

#ضمن <mutex>

ساكن ثابت حرف * ggml_backend_remoting_device_get_name(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_منزل_GPU(dev);

    // Return the prefixed name that was built once during initialization
    أرجع gpu->cached_device_info.name;
}

ساكن ثابت حرف * ggml_backend_remoting_device_get_description(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_منزل_GPU(dev);

    // Return the pre-cached description from the virtgpu structure
    أرجع gpu->cached_device_info.description;
}

ساكن تعداد ggml_backend_dev_type ggml_backend_remoting_device_get_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_منزل_GPU(dev);

    أرجع (تعداد ggml_backend_dev_type) gpu->cached_device_info.type;
}

ساكن فراغ ggml_backend_remoting_device_get_حافظہ(ggml_backend_dev_t dev, size_t * حرر, size_t * کل) {
    virtgpu * gpu = DEV_منزل_GPU(dev);

    *حرر  = gpu->cached_device_info.حافظہ_free;
    *کل = gpu->cached_device_info.حافظہ_کل;
}

ساكن منطقي ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
#إذا USE_ALWAYS_TRUE_SUPPORTS_OP == 1
    /* ggml-rpc cheats it like this */
    /* with the current implementation of serialize_tensor, the src/view aren't properly passed */
    بن_ورتیا(dev);
    بن_ورتیا(op);

    أرجع سچ;
#وإلا
    virtgpu * gpu = DEV_منزل_GPU(dev);

    أرجع apir_device_supports_op(gpu, op);
#جے_انت
}

ساكن منطقي ggml_backend_remoting_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    منطقي supported = buft->device == dev;

    أرجع supported;
}

ساكن منطقي ggml_backend_remoting_device_offload_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    بن_ورتیا(dev);
    بن_ورتیا(op);

    أرجع خطأ;
}

ساكن فراغ ggml_backend_remoting_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_remoting_device_get_name(dev);
    props->description = ggml_backend_remoting_device_get_description(dev);
    props->type        = ggml_backend_remoting_device_get_type(dev);
    ggml_backend_remoting_device_get_حافظہ(dev, &props->حافظہ_free, &props->حافظہ_کل);

    virtgpu * gpu = DEV_منزل_GPU(dev);
    apir_device_get_props(gpu, &props->caps.async, &props->caps.host_بفر, &props->caps.بفر_from_host_ptr,
                          &props->caps.events, &props->caps.mmap_support);

    props->caps.بفر_from_host_ptr = خطأ;
    props->caps.async                = خطأ;
    props->caps.events               = خطأ;
}

ggml_backend_بفر_type_t ggml_backend_remoting_device_get_بفر_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_منزل_GPU(dev);

    ساكن std::atomic<منطقي>        initialized = خطأ;
    ساكن ggml_backend_بفر_type buft;

    إذا (!initialized) {
        ساكن std::mutex           mutex;
        std::lock_guard<std::mutex> جندرا(mutex);

        إذا (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_بفر_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (فراغ *) gpu->cached_بفر_type.host_handle,
            };
            initialized = سچ;
        }
    }

    أرجع &buft;
}

ساكن ggml_backend_بفر_type_t ggml_backend_remoting_device_get_بفر_from_ptr_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_منزل_GPU(dev);

    ساكن std::atomic<منطقي>        initialized = خطأ;
    ساكن ggml_backend_بفر_type buft;

    إذا (!initialized) {
        ساكن std::mutex           mutex;
        std::lock_guard<std::mutex> جندرا(mutex);

        إذا (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_بفر_from_ptr_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (فراغ *) gpu->cached_بفر_type.host_handle,
            };
            initialized = سچ;
        }
    }

    أرجع &buft;
}

ساكن ggml_backend_بفر_t ggml_backend_remoting_device_بفر_from_ptr(ggml_backend_dev_t dev,
                                                                          فراغ *             ptr,
                                                                          size_t             size,
                                                                          size_t             max_tensor_ناپ) {
    virtgpu * gpu = DEV_منزل_GPU(dev);

    ggml_backend_remoting_بفر_context * context = (ggml_backend_remoting_بفر_context *) حافظہ_دیو(حجم(*context));
    إذا (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu          = gpu;
    context->apir_context = apir_device_بفر_from_ptr(gpu, size, max_tensor_ناپ);
    context->ادھار         = ptr;
    context->is_from_ptr  = سچ;

    ggml_backend_بفر_t بفر =
        ggml_backend_بفر_init(ggml_backend_remoting_device_get_بفر_from_ptr_type(dev),
                                 ggml_backend_remoting_بفر_from_ptr_interface, (فراغ *) context, size);

    أرجع بفر;
}

ثابت ggml_backend_device_i ggml_backend_remoting_device_interface = {
    /* .get_name             = */ ggml_backend_remoting_device_get_name,
    /* .get_description      = */ ggml_backend_remoting_device_get_description,
    /* .get_memory           = */ ggml_backend_remoting_device_get_حافظہ,
    /* .get_type             = */ ggml_backend_remoting_device_get_type,
    /* .get_props            = */ ggml_backend_remoting_device_get_props,
    /* .init_backend         = */ ggml_backend_remoting_device_init,
    /* .get_buffer_type      = */ ggml_backend_remoting_device_get_بفر_type,
    /* .get_host_buffer_type = */ عدم,
    /* .buffer_from_host_ptr = */ ggml_backend_remoting_device_بفر_from_ptr,
    /* .supports_op          = */ ggml_backend_remoting_device_supports_op,
    /* .supports_buft        = */ ggml_backend_remoting_device_supports_buft,
    /* .offload_op           = */ ggml_backend_remoting_device_offload_op,
    /* .event_new            = */ عدم,
    /* .event_free           = */ عدم,
    /* .event_synchronize    = */ عدم,
};
