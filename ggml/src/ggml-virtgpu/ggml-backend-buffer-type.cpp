#అన్తర్భావ "ggml-remoting.h"

స్థిర ggml_backend_బఫరు_t ggml_backend_remoting_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft,
                                                                            size_t                     size) {
    virtgpu * gpu = BUFT_గమ్యం_GPU(buft);

    ggml_backend_remoting_బఫరు_context * context = (ggml_backend_remoting_బఫరు_context *) జ్ఞాపకం_ఇవ్వు(పరిమాణం(*context));
    ఒకవేళ (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu = gpu;

    తార్కిక async_వాడని, host_బఫరు_వాడని, events_వాడని, mmap_support_వాడని;
    తార్కిక బఫరు_from_host_ptr;
    apir_device_get_props(gpu, &async_వాడని, &host_బఫరు_వాడని, &బఫరు_from_host_ptr, &events_వాడని, &mmap_support_వాడని);

    ఒకవేళ (బఫరు_from_host_ptr) {
        context->apir_context = apir_device_బఫరు_from_ptr(gpu, size, size);
        context->పునాది         = context->apir_context.shmem.mmap_ptr;
        context->is_from_ptr  = నిజం;
    } లేకపోతే {
        context->apir_context = apir_బఫరు_type_alloc_బఫరు(gpu, gpu->cached_బఫరు_type.host_handle, size);
        context->is_from_ptr  = అబద్ధం;
        context->పునాది         = ఖాళీ;
    }

    ggml_backend_బఫరు_t బఫరు =
        ggml_backend_బఫరు_init(buft, ggml_backend_remoting_బఫరు_interface, (శూన్య *) context, size);

    తిరిగి బఫరు;
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_remoting_బఫరు_type_get_name(ggml_backend_బఫరు_type_t buft) {
    virtgpu * gpu = BUFT_గమ్యం_GPU(buft);

    // Return the prefixed name that was built once during initialization
    తిరిగి gpu->cached_బఫరు_type.name;
}

స్థిర size_t ggml_backend_remoting_బఫరు_type_get_alignment(ggml_backend_బఫరు_type_t buft) {
    virtgpu * gpu = BUFT_గమ్యం_GPU(buft);

    తిరిగి gpu->cached_బఫరు_type.alignment;
}

స్థిర size_t ggml_backend_remoting_బఫరు_type_get_max_కొలత(ggml_backend_బఫరు_type_t buft) {
    virtgpu * gpu = BUFT_గమ్యం_GPU(buft);

    తిరిగి gpu->cached_బఫరు_type.max_size;
}

స్థిర size_t ggml_backend_remoting_బఫరు_type_get_alloc_కొలత(ggml_backend_బఫరు_type_t buft,
                                                               స్థిరరాశి ggml_tensor *        tensor) {
    virtgpu * gpu = BUFT_గమ్యం_GPU(buft);

    ఒకవేళ (tensor->బఫరు == ఖాళీ || !tensor->బఫరు->context ||
        !buft->device->iface.supports_buft(buft->device, tensor->బఫరు->buft)) {
        తిరిగి ggml_nbytes(tensor);
    }

    తిరిగి apir_బఫరు_type_get_alloc_కొలత(gpu, gpu->cached_బఫరు_type.host_handle, tensor);
}

స్థిరరాశి ggml_backend_బఫరు_type_i ggml_backend_remoting_బఫరు_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_బఫరు_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_remoting_బఫరు_type_alloc_బఫరు,
    /* .alloc_buffer_n      = */ ఖాళీ,
    /* .get_alignment       = */ ggml_backend_remoting_బఫరు_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_బఫరు_type_get_max_కొలత,
    /* .get_alloc_size      = */ ggml_backend_remoting_బఫరు_type_get_alloc_కొలత,
    /* .get_alloc_size_n    = */ ఖాళీ,
    /* .is_host             = */ ఖాళీ,
};

స్థిరరాశి ggml_backend_బఫరు_type_i ggml_backend_remoting_బఫరు_from_ptr_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_బఫరు_type_get_name,
    /* .alloc_buffer        = */ ఖాళీ,
    /* .alloc_buffer_n      = */ ఖాళీ,
    /* .get_alignment       = */ ggml_backend_remoting_బఫరు_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_బఫరు_type_get_max_కొలత,
    /* .get_alloc_size      = */ ggml_backend_remoting_బఫరు_type_get_alloc_కొలత,
    /* .get_alloc_size_n    = */ ఖాళీ,
    /* .is_host             = */ ఖాళీ,
};
