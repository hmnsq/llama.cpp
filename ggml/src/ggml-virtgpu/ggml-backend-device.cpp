#అన్తర్భావ "ggml-remoting.h"

#అన్తర్భావ <mutex>

స్థిర స్థిరరాశి అక్షరం * ggml_backend_remoting_device_get_name(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_గమ్యం_GPU(dev);

    // Return the prefixed name that was built once during initialization
    తిరిగి gpu->cached_device_info.name;
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_remoting_device_get_description(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_గమ్యం_GPU(dev);

    // Return the pre-cached description from the virtgpu structure
    తిరిగి gpu->cached_device_info.description;
}

స్థిర గణన ggml_backend_dev_type ggml_backend_remoting_device_get_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_గమ్యం_GPU(dev);

    తిరిగి (గణన ggml_backend_dev_type) gpu->cached_device_info.type;
}

స్థిర శూన్య ggml_backend_remoting_device_get_జ్ఞాపకం(ggml_backend_dev_t dev, size_t * విడుదల, size_t * మొత్తం) {
    virtgpu * gpu = DEV_గమ్యం_GPU(dev);

    *విడుదల  = gpu->cached_device_info.జ్ఞాపకం_free;
    *మొత్తం = gpu->cached_device_info.జ్ఞాపకం_మొత్తం;
}

స్థిర తార్కిక ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, స్థిరరాశి ggml_tensor * op) {
#ఒకవేళ USE_ALWAYS_TRUE_SUPPORTS_OP == 1
    /* ggml-rpc cheats it like this */
    /* with the current implementation of serialize_tensor, the src/view aren't properly passed */
    వాడని(dev);
    వాడని(op);

    తిరిగి నిజం;
#లేకపోతే
    virtgpu * gpu = DEV_గమ్యం_GPU(dev);

    తిరిగి apir_device_supports_op(gpu, op);
#అయితే_ముగింపు
}

స్థిర తార్కిక ggml_backend_remoting_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_బఫరు_type_t buft) {
    తార్కిక supported = buft->device == dev;

    తిరిగి supported;
}

స్థిర తార్కిక ggml_backend_remoting_device_offload_op(ggml_backend_dev_t dev, స్థిరరాశి ggml_tensor * op) {
    వాడని(dev);
    వాడని(op);

    తిరిగి అబద్ధం;
}

స్థిర శూన్య ggml_backend_remoting_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_remoting_device_get_name(dev);
    props->description = ggml_backend_remoting_device_get_description(dev);
    props->type        = ggml_backend_remoting_device_get_type(dev);
    ggml_backend_remoting_device_get_జ్ఞాపకం(dev, &props->జ్ఞాపకం_free, &props->జ్ఞాపకం_మొత్తం);

    virtgpu * gpu = DEV_గమ్యం_GPU(dev);
    apir_device_get_props(gpu, &props->caps.async, &props->caps.host_బఫరు, &props->caps.బఫరు_from_host_ptr,
                          &props->caps.events, &props->caps.mmap_support);

    props->caps.బఫరు_from_host_ptr = అబద్ధం;
    props->caps.async                = అబద్ధం;
    props->caps.events               = అబద్ధం;
}

ggml_backend_బఫరు_type_t ggml_backend_remoting_device_get_బఫరు_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_గమ్యం_GPU(dev);

    స్థిర std::atomic<తార్కిక>        initialized = అబద్ధం;
    స్థిర ggml_backend_బఫరు_type buft;

    ఒకవేళ (!initialized) {
        స్థిర std::mutex           mutex;
        std::lock_guard<std::mutex> తాళం(mutex);

        ఒకవేళ (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_బఫరు_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (శూన్య *) gpu->cached_బఫరు_type.host_handle,
            };
            initialized = నిజం;
        }
    }

    తిరిగి &buft;
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_remoting_device_get_బఫరు_from_ptr_type(ggml_backend_dev_t dev) {
    virtgpu * gpu = DEV_గమ్యం_GPU(dev);

    స్థిర std::atomic<తార్కిక>        initialized = అబద్ధం;
    స్థిర ggml_backend_బఫరు_type buft;

    ఒకవేళ (!initialized) {
        స్థిర std::mutex           mutex;
        std::lock_guard<std::mutex> తాళం(mutex);

        ఒకవేళ (!initialized) {
            buft = {
                /* .iface    = */ ggml_backend_remoting_బఫరు_from_ptr_type_interface,
                /* .device   = */ dev,
                /* .context  = */ (శూన్య *) gpu->cached_బఫరు_type.host_handle,
            };
            initialized = నిజం;
        }
    }

    తిరిగి &buft;
}

స్థిర ggml_backend_బఫరు_t ggml_backend_remoting_device_బఫరు_from_ptr(ggml_backend_dev_t dev,
                                                                          శూన్య *             ptr,
                                                                          size_t             size,
                                                                          size_t             max_tensor_కొలత) {
    virtgpu * gpu = DEV_గమ్యం_GPU(dev);

    ggml_backend_remoting_బఫరు_context * context = (ggml_backend_remoting_బఫరు_context *) జ్ఞాపకం_ఇవ్వు(పరిమాణం(*context));
    ఒకవేళ (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu          = gpu;
    context->apir_context = apir_device_బఫరు_from_ptr(gpu, size, max_tensor_కొలత);
    context->పునాది         = ptr;
    context->is_from_ptr  = నిజం;

    ggml_backend_బఫరు_t బఫరు =
        ggml_backend_బఫరు_init(ggml_backend_remoting_device_get_బఫరు_from_ptr_type(dev),
                                 ggml_backend_remoting_బఫరు_from_ptr_interface, (శూన్య *) context, size);

    తిరిగి బఫరు;
}

స్థిరరాశి ggml_backend_device_i ggml_backend_remoting_device_interface = {
    /* .get_name             = */ ggml_backend_remoting_device_get_name,
    /* .get_description      = */ ggml_backend_remoting_device_get_description,
    /* .get_memory           = */ ggml_backend_remoting_device_get_జ్ఞాపకం,
    /* .get_type             = */ ggml_backend_remoting_device_get_type,
    /* .get_props            = */ ggml_backend_remoting_device_get_props,
    /* .init_backend         = */ ggml_backend_remoting_device_init,
    /* .get_buffer_type      = */ ggml_backend_remoting_device_get_బఫరు_type,
    /* .get_host_buffer_type = */ ఖాళీ,
    /* .buffer_from_host_ptr = */ ggml_backend_remoting_device_బఫరు_from_ptr,
    /* .supports_op          = */ ggml_backend_remoting_device_supports_op,
    /* .supports_buft        = */ ggml_backend_remoting_device_supports_buft,
    /* .offload_op           = */ ggml_backend_remoting_device_offload_op,
    /* .event_new            = */ ఖాళీ,
    /* .event_free           = */ ఖాళీ,
    /* .event_synchronize    = */ ఖాళీ,
};
