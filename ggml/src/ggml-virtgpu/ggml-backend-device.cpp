#شامل "ggml-remoting.h"

#شامل <mutex>

ساکن ثابت حرف * ggml_backend_remoting_device_get_name(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_مقصد_GPU(dev);

    // Return the prefixed name that was built once during initialization
    واپس gpu->cached_device_info.name;
}

ساکن ثابت حرف * ggml_backend_remoting_device_get_description(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_مقصد_GPU(dev);

    // Return the pre-cached description from the virtgpu structure
    واپس gpu->cached_device_info.description;
}

ساکن شمار ggml_backend_dev_type ggml_backend_remoting_device_get_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_مقصد_GPU(dev);

    واپس (شمار ggml_backend_dev_type) gpu->cached_device_info.type;
}

ساکن خالی ggml_backend_remoting_device_get_حافظہ(ggml_backend_dev_t dev, size_t * آزاد, size_t * کل) {
    virtgpu * gpu = DEV_مقصد_GPU(dev);

    *آزاد  = gpu->cached_device_info.حافظہ_free;
    *کل = gpu->cached_device_info.حافظہ_کل;
}

ساکن منطقی ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
#اگر USE_ALWAYS_TRUE_SUPPORTS_OP == 1
    /* ggml-rpc cheats it like this */
    /* with the current implementation of serialize_tensor, the src/view aren't properly passed */
    غیر_مستعمل(dev);
    غیر_مستعمل(op);

    واپس سچ;
#ورنہ
    virtgpu * gpu = DEV_مقصد_GPU(dev);

    واپس apir_device_supports_op(gpu, op);
#اگر_ختم
}

ساکن منطقی ggml_backend_remoting_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    منطقی supported = buft->device == dev;

    واپس supported;
}

ساکن منطقی ggml_backend_remoting_device_offload_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    غیر_مستعمل(dev);
    غیر_مستعمل(op);

    واپس جھوٹ;
}

ساکن خالی ggml_backend_remoting_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_remoting_device_get_name(dev);
    props->description = ggml_backend_remoting_device_get_description(dev);
    props->type        = ggml_backend_remoting_device_get_type(dev);
    ggml_backend_remoting_device_get_حافظہ(dev, &props->حافظہ_free, &props->حافظہ_کل);

    virtgpu * gpu = DEV_مقصد_GPU(dev);
    apir_device_get_props(gpu, &props->caps.async, &props->caps.host_بفر, &props->caps.بفر_from_host_ptr,
                          &props->caps.events, &props->caps.mmap_support);

    props->caps.بفر_from_host_ptr = جھوٹ;
    props->caps.async                = جھوٹ;
    props->caps.events               = جھوٹ;
}

ggml_backend_بفر_type_t ggml_backend_remoting_device_get_بفر_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_مقصد_GPU(dev);

    ساکن std::atomic<منطقی>        initialized = جھوٹ;
    ساکن ggml_backend_بفر_type buft;

    اگر (!initialized) {
        ساکن std::mutex           mutex;
        std::lock_guard<std::mutex> تالا(mutex);

        اگر (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_بفر_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (خالی *) gpu->cached_بفر_type.host_handle,
            };
            initialized = سچ;
        }
    }

    واپس &buft;
}

ساکن ggml_backend_بفر_type_t ggml_backend_remoting_device_get_بفر_from_ptr_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_مقصد_GPU(dev);

    ساکن std::atomic<منطقی>        initialized = جھوٹ;
    ساکن ggml_backend_بفر_type buft;

    اگر (!initialized) {
        ساکن std::mutex           mutex;
        std::lock_guard<std::mutex> تالا(mutex);

        اگر (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_بفر_from_ptr_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (خالی *) gpu->cached_بفر_type.host_handle,
            };
            initialized = سچ;
        }
    }

    واپس &buft;
}

ساکن ggml_backend_بفر_t ggml_backend_remoting_device_بفر_from_ptr(ggml_backend_dev_t dev,
                                                                          خالی *             ptr,
                                                                          size_t             size,
                                                                          size_t             max_tensor_سائز) {
    virtgpu * gpu = DEV_مقصد_GPU(dev);

    ggml_backend_remoting_بفر_context * context = (ggml_backend_remoting_بفر_context *) حافظہ_دو(حجم(*context));
    اگر (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu          = gpu;
    context->apir_context = apir_device_بفر_from_ptr(gpu, size, max_tensor_سائز);
    context->بنیاد         = ptr;
    context->is_from_ptr  = سچ;

    ggml_backend_بفر_t بفر =
        ggml_backend_بفر_init(ggml_backend_remoting_device_get_بفر_from_ptr_type(dev),
                                 ggml_backend_remoting_بفر_from_ptr_interface, (خالی *) context, size);

    واپس بفر;
}

ثابت ggml_backend_device_i ggml_backend_remoting_device_interface = {
    /* .get_name             = */ ggml_backend_remoting_device_get_name,
    /* .get_description      = */ ggml_backend_remoting_device_get_description,
    /* .get_memory           = */ ggml_backend_remoting_device_get_حافظہ,
    /* .get_type             = */ ggml_backend_remoting_device_get_type,
    /* .get_props            = */ ggml_backend_remoting_device_get_props,
    /* .init_backend         = */ ggml_backend_remoting_device_init,
    /* .get_buffer_type      = */ ggml_backend_remoting_device_get_بفر_type,
    /* .get_host_buffer_type = */ تہی,
    /* .buffer_from_host_ptr = */ ggml_backend_remoting_device_بفر_from_ptr,
    /* .supports_op          = */ ggml_backend_remoting_device_supports_op,
    /* .supports_buft        = */ ggml_backend_remoting_device_supports_buft,
    /* .offload_op           = */ ggml_backend_remoting_device_offload_op,
    /* .event_new            = */ تہی,
    /* .event_free           = */ تہی,
    /* .event_synchronize    = */ تہی,
};
