#ಅನ್ತರ್ಭಾವ "ggml-remoting.h"

#ಅನ್ತರ್ಭಾವ <mutex>

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_remoting_device_get_name(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_ಗಮ್ಯ_GPU(dev);

    // Return the prefixed name that was built once during initialization
    ಹಿಂತಿರುಗಿ gpu->cached_device_info.name;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_remoting_device_get_description(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_ಗಮ್ಯ_GPU(dev);

    // Return the pre-cached description from the virtgpu structure
    ಹಿಂತಿರುಗಿ gpu->cached_device_info.description;
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_backend_dev_type ggml_backend_remoting_device_get_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_ಗಮ್ಯ_GPU(dev);

    ಹಿಂತಿರುಗಿ (ಎಣಿಕೆ ggml_backend_dev_type) gpu->cached_device_info.type;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_remoting_device_get_ಸ್ಮರಣೆ(ggml_backend_dev_t dev, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
    virtgpu * gpu = DEV_ಗಮ್ಯ_GPU(dev);

    *ಬಿಡುಗಡೆ  = gpu->cached_device_info.ಸ್ಮರಣೆ_free;
    *ಒಟ್ಟು = gpu->cached_device_info.ಸ್ಮರಣೆ_ಒಟ್ಟು;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
#ಆದರೆ USE_ALWAYS_TRUE_SUPPORTS_OP == 1
    /* ggml-rpc cheats it like this */
    /* with the current implementation of serialize_tensor, the src/view aren't properly passed */
    ಬಳಸದ(dev);
    ಬಳಸದ(op);

    ಹಿಂತಿರುಗಿ ನಿಜ;
#ಇಲ್ಲದಿದ್ದರೆ
    virtgpu * gpu = DEV_ಗಮ್ಯ_GPU(dev);

    ಹಿಂತಿರುಗಿ apir_device_supports_op(gpu, op);
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_remoting_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ಬಫರ್_type_t buft) {
    ತಾರ್ಕಿಕ supported = buft->device == dev;

    ಹಿಂತಿರುಗಿ supported;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_remoting_device_offload_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಬಳಸದ(dev);
    ಬಳಸದ(op);

    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_remoting_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_remoting_device_get_name(dev);
    props->description = ggml_backend_remoting_device_get_description(dev);
    props->type        = ggml_backend_remoting_device_get_type(dev);
    ggml_backend_remoting_device_get_ಸ್ಮರಣೆ(dev, &props->ಸ್ಮರಣೆ_free, &props->ಸ್ಮರಣೆ_ಒಟ್ಟು);

    virtgpu * gpu = DEV_ಗಮ್ಯ_GPU(dev);
    apir_device_get_props(gpu, &props->caps.async, &props->caps.host_ಬಫರ್, &props->caps.ಬಫರ್_from_host_ptr,
                          &props->caps.events, &props->caps.mmap_support);

    props->caps.ಬಫರ್_from_host_ptr = ಸುಳ್ಳು;
    props->caps.async                = ಸುಳ್ಳು;
    props->caps.events               = ಸುಳ್ಳು;
}

ggml_backend_ಬಫರ್_type_t ggml_backend_remoting_device_get_ಬಫರ್_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_ಗಮ್ಯ_GPU(dev);

    ಸ್ಥಿರ std::atomic<ತಾರ್ಕಿಕ>        initialized = ಸುಳ್ಳು;
    ಸ್ಥಿರ ggml_backend_ಬಫರ್_type buft;

    ಆದರೆ (!initialized) {
        ಸ್ಥಿರ std::mutex           mutex;
        std::lock_guard<std::mutex> ಬೀಗ(mutex);

        ಆದರೆ (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_ಬಫರ್_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (ಶೂನ್ಯ *) gpu->cached_ಬಫರ್_type.host_handle,
            };
            initialized = ನಿಜ;
        }
    }

    ಹಿಂತಿರುಗಿ &buft;
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_remoting_device_get_ಬಫರ್_from_ptr_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_ಗಮ್ಯ_GPU(dev);

    ಸ್ಥಿರ std::atomic<ತಾರ್ಕಿಕ>        initialized = ಸುಳ್ಳು;
    ಸ್ಥಿರ ggml_backend_ಬಫರ್_type buft;

    ಆದರೆ (!initialized) {
        ಸ್ಥಿರ std::mutex           mutex;
        std::lock_guard<std::mutex> ಬೀಗ(mutex);

        ಆದರೆ (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_ಬಫರ್_from_ptr_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (ಶೂನ್ಯ *) gpu->cached_ಬಫರ್_type.host_handle,
            };
            initialized = ನಿಜ;
        }
    }

    ಹಿಂತಿರುಗಿ &buft;
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_remoting_device_ಬಫರ್_from_ptr(ggml_backend_dev_t dev,
                                                                          ಶೂನ್ಯ *             ptr,
                                                                          size_t             size,
                                                                          size_t             max_tensor_ಗಾತ್ರ) {
    virtgpu * gpu = DEV_ಗಮ್ಯ_GPU(dev);

    ggml_backend_remoting_ಬಫರ್_context * context = (ggml_backend_remoting_ಬಫರ್_context *) ಸ್ಮರಣೆ_ಕೊಡು(ಗಾತ್ರ(*context));
    ಆದರೆ (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu          = gpu;
    context->apir_context = apir_device_ಬಫರ್_from_ptr(gpu, size, max_tensor_ಗಾತ್ರ);
    context->ತಳ         = ptr;
    context->is_from_ptr  = ನಿಜ;

    ggml_backend_ಬಫರ್_t ಬಫರ್ =
        ggml_backend_ಬಫರ್_init(ggml_backend_remoting_device_get_ಬಫರ್_from_ptr_type(dev),
                                 ggml_backend_remoting_ಬಫರ್_from_ptr_interface, (ಶೂನ್ಯ *) context, size);

    ಹಿಂತಿರುಗಿ ಬಫರ್;
}

ಸ್ಥಿರಾಂಕ ggml_backend_device_i ggml_backend_remoting_device_interface = {
    /* .get_name             = */ ggml_backend_remoting_device_get_name,
    /* .get_description      = */ ggml_backend_remoting_device_get_description,
    /* .get_memory           = */ ggml_backend_remoting_device_get_ಸ್ಮರಣೆ,
    /* .get_type             = */ ggml_backend_remoting_device_get_type,
    /* .get_props            = */ ggml_backend_remoting_device_get_props,
    /* .init_backend         = */ ggml_backend_remoting_device_init,
    /* .get_buffer_type      = */ ggml_backend_remoting_device_get_ಬಫರ್_type,
    /* .get_host_buffer_type = */ ಖಾಲಿ,
    /* .buffer_from_host_ptr = */ ggml_backend_remoting_device_ಬಫರ್_from_ptr,
    /* .supports_op          = */ ggml_backend_remoting_device_supports_op,
    /* .supports_buft        = */ ggml_backend_remoting_device_supports_buft,
    /* .offload_op           = */ ggml_backend_remoting_device_offload_op,
    /* .event_new            = */ ಖಾಲಿ,
    /* .event_free           = */ ಖಾಲಿ,
    /* .event_synchronize    = */ ಖಾಲಿ,
};
