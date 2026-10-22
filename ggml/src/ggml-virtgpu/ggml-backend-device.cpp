#अन्तर्भाव "ggml-remoting.h"

#अन्तर्भाव <mutex>

स्थिर ध्रुव वर्ण * ggml_backend_remoting_device_get_name(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_गन्तव्यम्_GPU(dev);

    // Return the prefixed name that was built once during initialization
    प्रत्यागम gpu->cached_device_info.name;
}

स्थिर ध्रुव वर्ण * ggml_backend_remoting_device_get_description(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_गन्तव्यम्_GPU(dev);

    // Return the pre-cached description from the virtgpu structure
    प्रत्यागम gpu->cached_device_info.description;
}

स्थिर गणना ggml_backend_dev_type ggml_backend_remoting_device_get_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_गन्तव्यम्_GPU(dev);

    प्रत्यागम (गणना ggml_backend_dev_type) gpu->cached_device_info.type;
}

स्थिर शून्य ggml_backend_remoting_device_get_स्मृतिः(ggml_backend_dev_t dev, size_t * मुक्त, size_t * योग) {
    virtgpu * gpu = DEV_गन्तव्यम्_GPU(dev);

    *मुक्त  = gpu->cached_device_info.स्मृतिः_free;
    *योग = gpu->cached_device_info.स्मृतिः_योग;
}

स्थिर तार्किक ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, ध्रुव ggml_tensor * op) {
#यदि USE_ALWAYS_TRUE_SUPPORTS_OP == 1
    /* ggml-rpc cheats it like this */
    /* with the current implementation of serialize_tensor, the src/view aren't properly passed */
    अप्रयुक्तम्(dev);
    अप्रयुक्तम्(op);

    प्रत्यागम सत्य;
#अन्यथा
    virtgpu * gpu = DEV_गन्तव्यम्_GPU(dev);

    प्रत्यागम apir_device_supports_op(gpu, op);
#यदि_अन्तः
}

स्थिर तार्किक ggml_backend_remoting_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    तार्किक supported = buft->device == dev;

    प्रत्यागम supported;
}

स्थिर तार्किक ggml_backend_remoting_device_offload_op(ggml_backend_dev_t dev, ध्रुव ggml_tensor * op) {
    अप्रयुक्तम्(dev);
    अप्रयुक्तम्(op);

    प्रत्यागम असत्य;
}

स्थिर शून्य ggml_backend_remoting_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_remoting_device_get_name(dev);
    props->description = ggml_backend_remoting_device_get_description(dev);
    props->type        = ggml_backend_remoting_device_get_type(dev);
    ggml_backend_remoting_device_get_स्मृतिः(dev, &props->स्मृतिः_free, &props->स्मृतिः_योग);

    virtgpu * gpu = DEV_गन्तव्यम्_GPU(dev);
    apir_device_get_props(gpu, &props->caps.async, &props->caps.host_बफर, &props->caps.बफर_from_host_ptr,
                          &props->caps.events, &props->caps.mmap_support);

    props->caps.बफर_from_host_ptr = असत्य;
    props->caps.async                = असत्य;
    props->caps.events               = असत्य;
}

ggml_backend_बफर_type_t ggml_backend_remoting_device_get_बफर_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_गन्तव्यम्_GPU(dev);

    स्थिर std::atomic<तार्किक>        initialized = असत्य;
    स्थिर ggml_backend_बफर_type buft;

    यदि (!initialized) {
        स्थिर std::mutex           mutex;
        std::lock_guard<std::mutex> तालकम्(mutex);

        यदि (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_बफर_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (शून्य *) gpu->cached_बफर_type.host_handle,
            };
            initialized = सत्य;
        }
    }

    प्रत्यागम &buft;
}

स्थिर ggml_backend_बफर_type_t ggml_backend_remoting_device_get_बफर_from_ptr_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_गन्तव्यम्_GPU(dev);

    स्थिर std::atomic<तार्किक>        initialized = असत्य;
    स्थिर ggml_backend_बफर_type buft;

    यदि (!initialized) {
        स्थिर std::mutex           mutex;
        std::lock_guard<std::mutex> तालकम्(mutex);

        यदि (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_बफर_from_ptr_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (शून्य *) gpu->cached_बफर_type.host_handle,
            };
            initialized = सत्य;
        }
    }

    प्रत्यागम &buft;
}

स्थिर ggml_backend_बफर_t ggml_backend_remoting_device_बफर_from_ptr(ggml_backend_dev_t dev,
                                                                          शून्य *             ptr,
                                                                          size_t             size,
                                                                          size_t             max_tensor_आकार) {
    virtgpu * gpu = DEV_गन्तव्यम्_GPU(dev);

    ggml_backend_remoting_बफर_context * context = (ggml_backend_remoting_बफर_context *) स्मृतिं_देहि(परिमाण(*context));
    यदि (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu          = gpu;
    context->apir_context = apir_device_बफर_from_ptr(gpu, size, max_tensor_आकार);
    context->आधारः         = ptr;
    context->is_from_ptr  = सत्य;

    ggml_backend_बफर_t बफर =
        ggml_backend_बफर_init(ggml_backend_remoting_device_get_बफर_from_ptr_type(dev),
                                 ggml_backend_remoting_बफर_from_ptr_interface, (शून्य *) context, size);

    प्रत्यागम बफर;
}

ध्रुव ggml_backend_device_i ggml_backend_remoting_device_interface = {
    /* .get_name             = */ ggml_backend_remoting_device_get_name,
    /* .get_description      = */ ggml_backend_remoting_device_get_description,
    /* .get_memory           = */ ggml_backend_remoting_device_get_स्मृतिः,
    /* .get_type             = */ ggml_backend_remoting_device_get_type,
    /* .get_props            = */ ggml_backend_remoting_device_get_props,
    /* .init_backend         = */ ggml_backend_remoting_device_init,
    /* .get_buffer_type      = */ ggml_backend_remoting_device_get_बफर_type,
    /* .get_host_buffer_type = */ रिक्त,
    /* .buffer_from_host_ptr = */ ggml_backend_remoting_device_बफर_from_ptr,
    /* .supports_op          = */ ggml_backend_remoting_device_supports_op,
    /* .supports_buft        = */ ggml_backend_remoting_device_supports_buft,
    /* .offload_op           = */ ggml_backend_remoting_device_offload_op,
    /* .event_new            = */ रिक्त,
    /* .event_free           = */ रिक्त,
    /* .event_synchronize    = */ रिक्त,
};
