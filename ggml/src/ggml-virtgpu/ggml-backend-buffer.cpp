#അന്തര്ഭാവ "ggml-remoting.h"

#define BUFFER_TO_GPU(name) ((ggml_backend_remoting_buffer_context *) (name)->context)->gpu

സ്ഥിതം ശൂന്യം * ggml_backend_remoting_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) buffer->context;
    എങ്കിൽ (context->base) {
        തിരികെ context->base;
    }

    context->base = apir_buffer_get_base(BUFFER_TO_GPU(buffer), BUFFER_TO_APIR_CONTEXT(buffer));

    തിരികെ context->base;
}

സ്ഥിതം ശൂന്യം ggml_backend_remoting_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                    ggml_tensor *         tensor,
                                                    സ്ഥിരം ശൂന്യം *          data,
                                                    size_t                offset,
                                                    size_t                size) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    എങ്കിൽ (context->is_from_ptr) {
        memcpy((അക്ഷരം *) tensor->data + offset, data, size);
    } അല്ലെങ്കിൽ {
        apir_buffer_set_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), tensor, data, offset, size);
    }

    തിരികെ;
}

സ്ഥിതം ശൂന്യം ggml_backend_remoting_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                    സ്ഥിരം ggml_tensor *   tensor,
                                                    ശൂന്യം *                data,
                                                    size_t                offset,
                                                    size_t                size) {
    virtgpu *                              gpu     = BUFFER_TO_GPU(buffer);
    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    എങ്കിൽ (context->is_from_ptr) {
        memcpy(data, (സ്ഥിരം അക്ഷരം *) tensor->data + offset, size);
    } അല്ലെങ്കിൽ {
        apir_buffer_get_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), tensor, data, offset, size);
    }
}

സ്ഥിതം ശൂന്യം ggml_backend_remoting_buffer_set_tensor_from_ptr(ggml_backend_buffer_t buffer,
                                                             ggml_tensor *         tensor,
                                                             സ്ഥിരം ശൂന്യം *          data,
                                                             size_t                offset,
                                                             size_t                size) {
    UNUSED(buffer);

    memcpy((അക്ഷരം *) tensor->data + offset, data, size);

    തിരികെ;
}

സ്ഥിതം ശൂന്യം ggml_backend_remoting_buffer_get_tensor_from_ptr(ggml_backend_buffer_t buffer,
                                                             സ്ഥിരം ggml_tensor *   tensor,
                                                             ശൂന്യം *                data,
                                                             size_t                offset,
                                                             size_t                size) {
    UNUSED(buffer);

    memcpy(data, (സ്ഥിരം അക്ഷരം *) tensor->data + offset, size);
}

സ്ഥിതം തർക്കം ggml_backend_remoting_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                    സ്ഥിരം ggml_tensor *   src,
                                                    ggml_tensor *         dst) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    തർക്കം ret = apir_buffer_cpy_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), src, dst);

    തിരികെ ret;
}

സ്ഥിതം ശൂന്യം ggml_backend_remoting_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    apir_buffer_clear(gpu, BUFFER_TO_APIR_CONTEXT(buffer), value);

    തിരികെ;
}

സ്ഥിതം ശൂന്യം ggml_backend_remoting_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    apir_buffer_free_buffer(gpu, BUFFER_TO_APIR_CONTEXT(buffer));

    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    സ്വതന്ത്രം(context);
    buffer->context = ശൂന്യ;
}

സ്ഥിരം ggml_backend_buffer_i ggml_backend_remoting_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_remoting_buffer_get_base,
    /* .init_tensor     = */ ശൂന്യ,
    /* .memset_tensor   = */ ശൂന്യ,
    /* .set_tensor      = */ ggml_backend_remoting_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_remoting_buffer_get_tensor,
    /* .set_tensor_2d   = */ ശൂന്യ,
    /* .get_tensor_2d   = */ ശൂന്യ,
    /* .cpy_tensor      = */ ggml_backend_remoting_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_buffer_clear,
    /* .reset           = */ ശൂന്യ,
};

സ്ഥിരം ggml_backend_buffer_i ggml_backend_remoting_buffer_from_ptr_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_remoting_buffer_get_base,
    /* .init_tensor     = */ ശൂന്യ,
    /* .memset_tensor   = */ ശൂന്യ,
    /* .set_tensor      = */ ggml_backend_remoting_buffer_set_tensor_from_ptr,
    /* .get_tensor      = */ ggml_backend_remoting_buffer_get_tensor_from_ptr,
    /* .set_tensor_2d   = */ ശൂന്യ,
    /* .get_tensor_2d   = */ ശൂന്യ,
    /* .cpy_tensor      = */ ggml_backend_remoting_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_buffer_clear,
    /* .reset           = */ ശൂന്യ,
};
