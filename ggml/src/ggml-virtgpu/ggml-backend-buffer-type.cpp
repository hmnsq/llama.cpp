#অন্তর্ভাব "ggml-remoting.h"

স্থির ggml_backend_বাফার_t ggml_backend_remoting_বাফার_type_alloc_বাফার(ggml_backend_বাফার_type_t buft,
                                                                            size_t                     size) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    ggml_backend_remoting_বাফার_context * context = (ggml_backend_remoting_বাফার_context *) malloc(আকার(*context));
    যদি (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu = gpu;

    যৌক্তিক async__unused, host_বাফার_unused, events__unused, mmap_support__unused;
    যৌক্তিক বাফার_from_host_ptr;
    apir_device_get_props(gpu, &async__unused, &host_বাফার_unused, &বাফার_from_host_ptr, &events__unused, &mmap_support__unused);

    যদি (বাফার_from_host_ptr) {
        context->apir_context = apir_device_বাফার_from_ptr(gpu, size, size);
        context->base         = context->apir_context.shmem.mmap_ptr;
        context->is_from_ptr  = সত্য;
    } অন্যথা {
        context->apir_context = apir_বাফার_type_alloc_বাফার(gpu, gpu->cached_বাফার_type.host_handle, size);
        context->is_from_ptr  = মিথ্যা;
        context->base         = ফাঁকা;
    }

    ggml_backend_বাফার_t বাফার =
        ggml_backend_বাফার_init(buft, ggml_backend_remoting_বাফার_interface, (শূন্য *) context, size);

    ফেরত বাফার;
}

স্থির ধ্রুবক অক্ষর * ggml_backend_remoting_বাফার_type_get_name(ggml_backend_বাফার_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    // Return the prefixed name that was built once during initialization
    ফেরত gpu->cached_বাফার_type.name;
}

স্থির size_t ggml_backend_remoting_বাফার_type_get_alignment(ggml_backend_বাফার_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    ফেরত gpu->cached_বাফার_type.alignment;
}

স্থির size_t ggml_backend_remoting_বাফার_type_get_max_আকার(ggml_backend_বাফার_type_t buft) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    ফেরত gpu->cached_বাফার_type.max_size;
}

স্থির size_t ggml_backend_remoting_বাফার_type_get_alloc_আকার(ggml_backend_বাফার_type_t buft,
                                                               ধ্রুবক ggml_tensor *        tensor) {
    virtgpu * gpu = BUFT_TO_GPU(buft);

    যদি (tensor->বাফার == ফাঁকা || !tensor->বাফার->context ||
        !buft->device->iface.supports_buft(buft->device, tensor->বাফার->buft)) {
        ফেরত ggml_nbytes(tensor);
    }

    ফেরত apir_বাফার_type_get_alloc_আকার(gpu, gpu->cached_বাফার_type.host_handle, tensor);
}

ধ্রুবক ggml_backend_বাফার_type_i ggml_backend_remoting_বাফার_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_বাফার_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_remoting_বাফার_type_alloc_বাফার,
    /* .alloc_buffer_n      = */ ফাঁকা,
    /* .get_alignment       = */ ggml_backend_remoting_বাফার_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_বাফার_type_get_max_আকার,
    /* .get_alloc_size      = */ ggml_backend_remoting_বাফার_type_get_alloc_আকার,
    /* .get_alloc_size_n    = */ ফাঁকা,
    /* .is_host             = */ ফাঁকা,
};

ধ্রুবক ggml_backend_বাফার_type_i ggml_backend_remoting_বাফার_from_ptr_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_বাফার_type_get_name,
    /* .alloc_buffer        = */ ফাঁকা,
    /* .alloc_buffer_n      = */ ফাঁকা,
    /* .get_alignment       = */ ggml_backend_remoting_বাফার_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_বাফার_type_get_max_আকার,
    /* .get_alloc_size      = */ ggml_backend_remoting_বাফার_type_get_alloc_আকার,
    /* .get_alloc_size_n    = */ ফাঁকা,
    /* .is_host             = */ ফাঁকা,
};
