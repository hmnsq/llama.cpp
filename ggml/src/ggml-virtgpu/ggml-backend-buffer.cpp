#ଅନ୍ତର୍ଭାଵ "ggml-remoting.h"

#define BUFFER_TO_GPU(name) ((ggml_backend_remoting_buffer_context *) (name)->context)->gpu

ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_remoting_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) buffer->context;
    ଯଦି (context->base) {
        ଫେରନ୍ତୁ context->base;
    }

    context->base = apir_buffer_get_base(BUFFER_TO_GPU(buffer), BUFFER_TO_APIR_CONTEXT(buffer));

    ଫେରନ୍ତୁ context->base;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_remoting_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                    ggml_tensor *         tensor,
                                                    ଅଚଳ ଶୂନ୍ୟ *          data,
                                                    size_t                offset,
                                                    size_t                size) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    ଯଦି (context->is_from_ptr) {
        memcpy((ଅକ୍ଷର *) tensor->data + offset, data, size);
    } ଅନ୍ୟଥା {
        apir_buffer_set_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), tensor, data, offset, size);
    }

    ଫେରନ୍ତୁ;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_remoting_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                    ଅଚଳ ggml_tensor *   tensor,
                                                    ଶୂନ୍ୟ *                data,
                                                    size_t                offset,
                                                    size_t                size) {
    virtgpu *                              gpu     = BUFFER_TO_GPU(buffer);
    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    ଯଦି (context->is_from_ptr) {
        memcpy(data, (ଅଚଳ ଅକ୍ଷର *) tensor->data + offset, size);
    } ଅନ୍ୟଥା {
        apir_buffer_get_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), tensor, data, offset, size);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_remoting_buffer_set_tensor_from_ptr(ggml_backend_buffer_t buffer,
                                                             ggml_tensor *         tensor,
                                                             ଅଚଳ ଶୂନ୍ୟ *          data,
                                                             size_t                offset,
                                                             size_t                size) {
    UNUSED(buffer);

    memcpy((ଅକ୍ଷର *) tensor->data + offset, data, size);

    ଫେରନ୍ତୁ;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_remoting_buffer_get_tensor_from_ptr(ggml_backend_buffer_t buffer,
                                                             ଅଚଳ ggml_tensor *   tensor,
                                                             ଶୂନ୍ୟ *                data,
                                                             size_t                offset,
                                                             size_t                size) {
    UNUSED(buffer);

    memcpy(data, (ଅଚଳ ଅକ୍ଷର *) tensor->data + offset, size);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_remoting_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                    ଅଚଳ ggml_tensor *   src,
                                                    ggml_tensor *         dst) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    ତାର୍କିକ ret = apir_buffer_cpy_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), src, dst);

    ଫେରନ୍ତୁ ret;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_remoting_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    apir_buffer_clear(gpu, BUFFER_TO_APIR_CONTEXT(buffer), value);

    ଫେରନ୍ତୁ;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_remoting_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    apir_buffer_free_buffer(gpu, BUFFER_TO_APIR_CONTEXT(buffer));

    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    ମୁକ୍ତ(context);
    buffer->context = ଖାଲି;
}

ଅଚଳ ggml_backend_buffer_i ggml_backend_remoting_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_remoting_buffer_get_base,
    /* .init_tensor     = */ ଖାଲି,
    /* .memset_tensor   = */ ଖାଲି,
    /* .set_tensor      = */ ggml_backend_remoting_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_remoting_buffer_get_tensor,
    /* .set_tensor_2d   = */ ଖାଲି,
    /* .get_tensor_2d   = */ ଖାଲି,
    /* .cpy_tensor      = */ ggml_backend_remoting_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_buffer_clear,
    /* .reset           = */ ଖାଲି,
};

ଅଚଳ ggml_backend_buffer_i ggml_backend_remoting_buffer_from_ptr_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_remoting_buffer_get_base,
    /* .init_tensor     = */ ଖାଲି,
    /* .memset_tensor   = */ ଖାଲି,
    /* .set_tensor      = */ ggml_backend_remoting_buffer_set_tensor_from_ptr,
    /* .get_tensor      = */ ggml_backend_remoting_buffer_get_tensor_from_ptr,
    /* .set_tensor_2d   = */ ଖାଲି,
    /* .get_tensor_2d   = */ ଖାଲି,
    /* .cpy_tensor      = */ ggml_backend_remoting_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_buffer_clear,
    /* .reset           = */ ଖାଲି,
};
