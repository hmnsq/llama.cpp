#ಅನ್ತರ್ಭಾವ "ggml-remoting.h"

#ವ್ಯಾಖ್ಯಾನಿಸು ಬಫರ್_ಗಮ್ಯ_GPU(name) ((ggml_backend_remoting_ಬಫರ್_context *) (name)->context)->gpu

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_remoting_ಬಫರ್_get_ತಳ(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ggml_backend_remoting_ಬಫರ್_context * context = (ggml_backend_remoting_ಬಫರ್_context *) ಬಫರ್->context;
    ಆದರೆ (context->ತಳ) {
        ಹಿಂತಿರುಗಿ context->ತಳ;
    }

    context->ತಳ = apir_ಬಫರ್_get_ತಳ(ಬಫರ್_ಗಮ್ಯ_GPU(ಬಫರ್), ಬಫರ್_ಗಮ್ಯ_APIR_CONTEXT(ಬಫರ್));

    ಹಿಂತಿರುಗಿ context->ತಳ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_remoting_ಬಫರ್_ಗಣ_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                    ggml_tensor *         tensor,
                                                    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *          ದತ್ತಾಂಶ,
                                                    size_t                ಸ್ಥಾನಾಂತರ,
                                                    size_t                size) {
    virtgpu * gpu = ಬಫರ್_ಗಮ್ಯ_GPU(ಬಫರ್);

    ggml_backend_remoting_ಬಫರ್_context * context = ಬಫರ್_ಗಮ್ಯ_GGML_CONTEXT(ಬಫರ್);
    ಆದರೆ (context->is_from_ptr) {
        ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, size);
    } ಇಲ್ಲದಿದ್ದರೆ {
        apir_ಬಫರ್_ಗಣ_tensor(gpu, ಬಫರ್_ಗಮ್ಯ_APIR_CONTEXT(ಬಫರ್), tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
    }

    ಹಿಂತಿರುಗಿ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_remoting_ಬಫರ್_get_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                    ಸ್ಥಿರಾಂಕ ggml_tensor *   tensor,
                                                    ಶೂನ್ಯ *                ದತ್ತಾಂಶ,
                                                    size_t                ಸ್ಥಾನಾಂತರ,
                                                    size_t                size) {
    virtgpu *                              gpu     = ಬಫರ್_ಗಮ್ಯ_GPU(ಬಫರ್);
    ggml_backend_remoting_ಬಫರ್_context * context = ಬಫರ್_ಗಮ್ಯ_GGML_CONTEXT(ಬಫರ್);
    ಆದರೆ (context->is_from_ptr) {
        ಸ್ಮರಣೆ_ನಕಲು(ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, size);
    } ಇಲ್ಲದಿದ್ದರೆ {
        apir_ಬಫರ್_get_tensor(gpu, ಬಫರ್_ಗಮ್ಯ_APIR_CONTEXT(ಬಫರ್), tensor, ದತ್ತಾಂಶ, ಸ್ಥಾನಾಂತರ, size);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_remoting_ಬಫರ್_ಗಣ_tensor_from_ptr(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                             ggml_tensor *         tensor,
                                                             ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *          ದತ್ತಾಂಶ,
                                                             size_t                ಸ್ಥಾನಾಂತರ,
                                                             size_t                size) {
    ಬಳಸದ(ಬಫರ್);

    ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, size);

    ಹಿಂತಿರುಗಿ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_remoting_ಬಫರ್_get_tensor_from_ptr(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                             ಸ್ಥಿರಾಂಕ ggml_tensor *   tensor,
                                                             ಶೂನ್ಯ *                ದತ್ತಾಂಶ,
                                                             size_t                ಸ್ಥಾನಾಂತರ,
                                                             size_t                size) {
    ಬಳಸದ(ಬಫರ್);

    ಸ್ಮರಣೆ_ನಕಲು(ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_remoting_ಬಫರ್_cpy_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                    ಸ್ಥಿರಾಂಕ ggml_tensor *   src,
                                                    ggml_tensor *         dst) {
    virtgpu * gpu = ಬಫರ್_ಗಮ್ಯ_GPU(ಬಫರ್);

    ತಾರ್ಕಿಕ ret = apir_ಬಫರ್_cpy_tensor(gpu, ಬಫರ್_ಗಮ್ಯ_APIR_CONTEXT(ಬಫರ್), src, dst);

    ಹಿಂತಿರುಗಿ ret;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_remoting_ಬಫರ್_clear(ggml_backend_ಬಫರ್_t ಬಫರ್, uint8_t value) {
    virtgpu * gpu = ಬಫರ್_ಗಮ್ಯ_GPU(ಬಫರ್);

    apir_ಬಫರ್_clear(gpu, ಬಫರ್_ಗಮ್ಯ_APIR_CONTEXT(ಬಫರ್), value);

    ಹಿಂತಿರುಗಿ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_remoting_ಬಫರ್_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    virtgpu * gpu = ಬಫರ್_ಗಮ್ಯ_GPU(ಬಫರ್);

    apir_ಬಫರ್_free_ಬಫರ್(gpu, ಬಫರ್_ಗಮ್ಯ_APIR_CONTEXT(ಬಫರ್));

    ggml_backend_remoting_ಬಫರ್_context * context = ಬಫರ್_ಗಮ್ಯ_GGML_CONTEXT(ಬಫರ್);
    ಬಿಡುಗಡೆ(context);
    ಬಫರ್->context = ಖಾಲಿ;
}

ಸ್ಥಿರಾಂಕ ggml_backend_ಬಫರ್_i ggml_backend_remoting_ಬಫರ್_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_ಬಫರ್_free_ಬಫರ್,
    /* .get_base        = */ ggml_backend_remoting_ಬಫರ್_get_ತಳ,
    /* .init_tensor     = */ ಖಾಲಿ,
    /* .memset_tensor   = */ ಖಾಲಿ,
    /* .set_tensor      = */ ggml_backend_remoting_ಬಫರ್_ಗಣ_tensor,
    /* .get_tensor      = */ ggml_backend_remoting_ಬಫರ್_get_tensor,
    /* .set_tensor_2d   = */ ಖಾಲಿ,
    /* .get_tensor_2d   = */ ಖಾಲಿ,
    /* .cpy_tensor      = */ ggml_backend_remoting_ಬಫರ್_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_ಬಫರ್_clear,
    /* .reset           = */ ಖಾಲಿ,
};

ಸ್ಥಿರಾಂಕ ggml_backend_ಬಫರ್_i ggml_backend_remoting_ಬಫರ್_from_ptr_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_ಬಫರ್_free_ಬಫರ್,
    /* .get_base        = */ ggml_backend_remoting_ಬಫರ್_get_ತಳ,
    /* .init_tensor     = */ ಖಾಲಿ,
    /* .memset_tensor   = */ ಖಾಲಿ,
    /* .set_tensor      = */ ggml_backend_remoting_ಬಫರ್_ಗಣ_tensor_from_ptr,
    /* .get_tensor      = */ ggml_backend_remoting_ಬಫರ್_get_tensor_from_ptr,
    /* .set_tensor_2d   = */ ಖಾಲಿ,
    /* .get_tensor_2d   = */ ಖಾಲಿ,
    /* .cpy_tensor      = */ ggml_backend_remoting_ಬಫರ್_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_ಬಫರ್_clear,
    /* .reset           = */ ಖಾಲಿ,
};
