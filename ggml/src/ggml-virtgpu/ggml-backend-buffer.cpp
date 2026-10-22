#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-remoting.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਬਫਰ_ਮੰਜ਼ਿਲ_GPU(name) ((ggml_backend_remoting_ਬਫਰ_context *) (name)->context)->gpu

ਸਥਿਰ ਖਾਲੀ * ggml_backend_remoting_ਬਫਰ_get_ਅਧਾਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ggml_backend_remoting_ਬਫਰ_context * context = (ggml_backend_remoting_ਬਫਰ_context *) ਬਫਰ->context;
    ਜੇ (context->ਅਧਾਰ) {
        ਵਾਪਸ context->ਅਧਾਰ;
    }

    context->ਅਧਾਰ = apir_ਬਫਰ_get_ਅਧਾਰ(ਬਫਰ_ਮੰਜ਼ਿਲ_GPU(ਬਫਰ), ਬਫਰ_ਮੰਜ਼ਿਲ_APIR_CONTEXT(ਬਫਰ));

    ਵਾਪਸ context->ਅਧਾਰ;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_remoting_ਬਫਰ_ਸੈੱਟ_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                    ggml_tensor *         tensor,
                                                    ਅਚਲ ਖਾਲੀ *          ਅੰਕੜੇ,
                                                    size_t                ਵਿੱਥ,
                                                    size_t                size) {
    virtgpu * gpu = ਬਫਰ_ਮੰਜ਼ਿਲ_GPU(ਬਫਰ);

    ggml_backend_remoting_ਬਫਰ_context * context = ਬਫਰ_ਮੰਜ਼ਿਲ_GGML_CONTEXT(ਬਫਰ);
    ਜੇ (context->is_from_ptr) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, ਅੰਕੜੇ, size);
    } ਨਹੀਂਤਾਂ {
        apir_ਬਫਰ_ਸੈੱਟ_tensor(gpu, ਬਫਰ_ਮੰਜ਼ਿਲ_APIR_CONTEXT(ਬਫਰ), tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
    }

    ਵਾਪਸ;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_remoting_ਬਫਰ_get_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                    ਅਚਲ ggml_tensor *   tensor,
                                                    ਖਾਲੀ *                ਅੰਕੜੇ,
                                                    size_t                ਵਿੱਥ,
                                                    size_t                size) {
    virtgpu *                              gpu     = ਬਫਰ_ਮੰਜ਼ਿਲ_GPU(ਬਫਰ);
    ggml_backend_remoting_ਬਫਰ_context * context = ਬਫਰ_ਮੰਜ਼ਿਲ_GGML_CONTEXT(ਬਫਰ);
    ਜੇ (context->is_from_ptr) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਅੰਕੜੇ, (ਅਚਲ ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, size);
    } ਨਹੀਂਤਾਂ {
        apir_ਬਫਰ_get_tensor(gpu, ਬਫਰ_ਮੰਜ਼ਿਲ_APIR_CONTEXT(ਬਫਰ), tensor, ਅੰਕੜੇ, ਵਿੱਥ, size);
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_remoting_ਬਫਰ_ਸੈੱਟ_tensor_from_ptr(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                             ggml_tensor *         tensor,
                                                             ਅਚਲ ਖਾਲੀ *          ਅੰਕੜੇ,
                                                             size_t                ਵਿੱਥ,
                                                             size_t                size) {
    ਅਣਵਰਤਿਆ(ਬਫਰ);

    ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, ਅੰਕੜੇ, size);

    ਵਾਪਸ;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_remoting_ਬਫਰ_get_tensor_from_ptr(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                             ਅਚਲ ggml_tensor *   tensor,
                                                             ਖਾਲੀ *                ਅੰਕੜੇ,
                                                             size_t                ਵਿੱਥ,
                                                             size_t                size) {
    ਅਣਵਰਤਿਆ(ਬਫਰ);

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਅੰਕੜੇ, (ਅਚਲ ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, size);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_remoting_ਬਫਰ_cpy_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                    ਅਚਲ ggml_tensor *   src,
                                                    ggml_tensor *         dst) {
    virtgpu * gpu = ਬਫਰ_ਮੰਜ਼ਿਲ_GPU(ਬਫਰ);

    ਤਾਰਕਿਕ ret = apir_ਬਫਰ_cpy_tensor(gpu, ਬਫਰ_ਮੰਜ਼ਿਲ_APIR_CONTEXT(ਬਫਰ), src, dst);

    ਵਾਪਸ ret;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_remoting_ਬਫਰ_clear(ggml_backend_ਬਫਰ_t ਬਫਰ, uint8_t value) {
    virtgpu * gpu = ਬਫਰ_ਮੰਜ਼ਿਲ_GPU(ਬਫਰ);

    apir_ਬਫਰ_clear(gpu, ਬਫਰ_ਮੰਜ਼ਿਲ_APIR_CONTEXT(ਬਫਰ), value);

    ਵਾਪਸ;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_remoting_ਬਫਰ_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    virtgpu * gpu = ਬਫਰ_ਮੰਜ਼ਿਲ_GPU(ਬਫਰ);

    apir_ਬਫਰ_free_ਬਫਰ(gpu, ਬਫਰ_ਮੰਜ਼ਿਲ_APIR_CONTEXT(ਬਫਰ));

    ggml_backend_remoting_ਬਫਰ_context * context = ਬਫਰ_ਮੰਜ਼ਿਲ_GGML_CONTEXT(ਬਫਰ);
    ਮੁਕਤ(context);
    ਬਫਰ->context = ਨਿਰਾ;
}

ਅਚਲ ggml_backend_ਬਫਰ_i ggml_backend_remoting_ਬਫਰ_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_ਬਫਰ_free_ਬਫਰ,
    /* .get_base        = */ ggml_backend_remoting_ਬਫਰ_get_ਅਧਾਰ,
    /* .init_tensor     = */ ਨਿਰਾ,
    /* .memset_tensor   = */ ਨਿਰਾ,
    /* .set_tensor      = */ ggml_backend_remoting_ਬਫਰ_ਸੈੱਟ_tensor,
    /* .get_tensor      = */ ggml_backend_remoting_ਬਫਰ_get_tensor,
    /* .set_tensor_2d   = */ ਨਿਰਾ,
    /* .get_tensor_2d   = */ ਨਿਰਾ,
    /* .cpy_tensor      = */ ggml_backend_remoting_ਬਫਰ_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_ਬਫਰ_clear,
    /* .reset           = */ ਨਿਰਾ,
};

ਅਚਲ ggml_backend_ਬਫਰ_i ggml_backend_remoting_ਬਫਰ_from_ptr_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_ਬਫਰ_free_ਬਫਰ,
    /* .get_base        = */ ggml_backend_remoting_ਬਫਰ_get_ਅਧਾਰ,
    /* .init_tensor     = */ ਨਿਰਾ,
    /* .memset_tensor   = */ ਨਿਰਾ,
    /* .set_tensor      = */ ggml_backend_remoting_ਬਫਰ_ਸੈੱਟ_tensor_from_ptr,
    /* .get_tensor      = */ ggml_backend_remoting_ਬਫਰ_get_tensor_from_ptr,
    /* .set_tensor_2d   = */ ਨਿਰਾ,
    /* .get_tensor_2d   = */ ਨਿਰਾ,
    /* .cpy_tensor      = */ ggml_backend_remoting_ਬਫਰ_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_ਬਫਰ_clear,
    /* .reset           = */ ਨਿਰਾ,
};
