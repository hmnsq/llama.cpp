#ಅನ್ತರ್ಭಾವ "ggml-remoting.h"

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_remoting_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft,
                                                                            size_t                     size) {
    virtgpu * gpu = BUFT_ಗಮ್ಯ_GPU(buft);

    ggml_backend_remoting_ಬಫರ್_context * context = (ggml_backend_remoting_ಬಫರ್_context *) ಸ್ಮರಣೆ_ಕೊಡು(ಗಾತ್ರ(*context));
    ಆದರೆ (!context) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the buffer context ...", __func__);
    }

    context->gpu = gpu;

    ತಾರ್ಕಿಕ async_ಬಳಸದ, host_ಬಫರ್_ಬಳಸದ, events_ಬಳಸದ, mmap_support_ಬಳಸದ;
    ತಾರ್ಕಿಕ ಬಫರ್_from_host_ptr;
    apir_device_get_props(gpu, &async_ಬಳಸದ, &host_ಬಫರ್_ಬಳಸದ, &ಬಫರ್_from_host_ptr, &events_ಬಳಸದ, &mmap_support_ಬಳಸದ);

    ಆದರೆ (ಬಫರ್_from_host_ptr) {
        context->apir_context = apir_device_ಬಫರ್_from_ptr(gpu, size, size);
        context->ತಳ         = context->apir_context.shmem.mmap_ptr;
        context->is_from_ptr  = ನಿಜ;
    } ಇಲ್ಲದಿದ್ದರೆ {
        context->apir_context = apir_ಬಫರ್_type_alloc_ಬಫರ್(gpu, gpu->cached_ಬಫರ್_type.host_handle, size);
        context->is_from_ptr  = ಸುಳ್ಳು;
        context->ತಳ         = ಖಾಲಿ;
    }

    ggml_backend_ಬಫರ್_t ಬಫರ್ =
        ggml_backend_ಬಫರ್_init(buft, ggml_backend_remoting_ಬಫರ್_interface, (ಶೂನ್ಯ *) context, size);

    ಹಿಂತಿರುಗಿ ಬಫರ್;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_remoting_ಬಫರ್_type_get_name(ggml_backend_ಬಫರ್_type_t buft) {
    virtgpu * gpu = BUFT_ಗಮ್ಯ_GPU(buft);

    // Return the prefixed name that was built once during initialization
    ಹಿಂತಿರುಗಿ gpu->cached_ಬಫರ್_type.name;
}

ಸ್ಥಿರ size_t ggml_backend_remoting_ಬಫರ್_type_get_alignment(ggml_backend_ಬಫರ್_type_t buft) {
    virtgpu * gpu = BUFT_ಗಮ್ಯ_GPU(buft);

    ಹಿಂತಿರುಗಿ gpu->cached_ಬಫರ್_type.alignment;
}

ಸ್ಥಿರ size_t ggml_backend_remoting_ಬಫರ್_type_get_max_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft) {
    virtgpu * gpu = BUFT_ಗಮ್ಯ_GPU(buft);

    ಹಿಂತಿರುಗಿ gpu->cached_ಬಫರ್_type.max_size;
}

ಸ್ಥಿರ size_t ggml_backend_remoting_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft,
                                                               ಸ್ಥಿರಾಂಕ ggml_tensor *        tensor) {
    virtgpu * gpu = BUFT_ಗಮ್ಯ_GPU(buft);

    ಆದರೆ (tensor->ಬಫರ್ == ಖಾಲಿ || !tensor->ಬಫರ್->context ||
        !buft->device->iface.supports_buft(buft->device, tensor->ಬಫರ್->buft)) {
        ಹಿಂತಿರುಗಿ ggml_nbytes(tensor);
    }

    ಹಿಂತಿರುಗಿ apir_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(gpu, gpu->cached_ಬಫರ್_type.host_handle, tensor);
}

ಸ್ಥಿರಾಂಕ ggml_backend_ಬಫರ್_type_i ggml_backend_remoting_ಬಫರ್_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_ಬಫರ್_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_remoting_ಬಫರ್_type_alloc_ಬಫರ್,
    /* .alloc_buffer_n      = */ ಖಾಲಿ,
    /* .get_alignment       = */ ggml_backend_remoting_ಬಫರ್_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_ಬಫರ್_type_get_max_ಗಾತ್ರ,
    /* .get_alloc_size      = */ ggml_backend_remoting_ಬಫರ್_type_get_alloc_ಗಾತ್ರ,
    /* .get_alloc_size_n    = */ ಖಾಲಿ,
    /* .is_host             = */ ಖಾಲಿ,
};

ಸ್ಥಿರಾಂಕ ggml_backend_ಬಫರ್_type_i ggml_backend_remoting_ಬಫರ್_from_ptr_type_interface = {
    /* .get_name            = */ ggml_backend_remoting_ಬಫರ್_type_get_name,
    /* .alloc_buffer        = */ ಖಾಲಿ,
    /* .alloc_buffer_n      = */ ಖಾಲಿ,
    /* .get_alignment       = */ ggml_backend_remoting_ಬಫರ್_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_remoting_ಬಫರ್_type_get_max_ಗಾತ್ರ,
    /* .get_alloc_size      = */ ggml_backend_remoting_ಬಫರ್_type_get_alloc_ಗಾತ್ರ,
    /* .get_alloc_size_n    = */ ಖಾಲಿ,
    /* .is_host             = */ ಖಾಲಿ,
};
