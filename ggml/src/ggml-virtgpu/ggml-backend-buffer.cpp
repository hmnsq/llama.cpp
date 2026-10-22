#અન્તર્ભાવ "ggml-remoting.h"

#define BUFFER_TO_GPU(name) ((ggml_backend_remoting_buffer_context *) (name)->context)->gpu

સ્થિર શૂન્ય * ggml_backend_remoting_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) buffer->context;
    જો (context->base) {
        પાછા context->base;
    }

    context->base = apir_buffer_get_base(BUFFER_TO_GPU(buffer), BUFFER_TO_APIR_CONTEXT(buffer));

    પાછા context->base;
}

સ્થિર શૂન્ય ggml_backend_remoting_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                    ggml_tensor *         tensor,
                                                    અચલ શૂન્ય *          data,
                                                    size_t                offset,
                                                    size_t                size) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    જો (context->is_from_ptr) {
        memcpy((અક્ષર *) tensor->data + offset, data, size);
    } નહીંતર {
        apir_buffer_set_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), tensor, data, offset, size);
    }

    પાછા;
}

સ્થિર શૂન્ય ggml_backend_remoting_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                    અચલ ggml_tensor *   tensor,
                                                    શૂન્ય *                data,
                                                    size_t                offset,
                                                    size_t                size) {
    virtgpu *                              gpu     = BUFFER_TO_GPU(buffer);
    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    જો (context->is_from_ptr) {
        memcpy(data, (અચલ અક્ષર *) tensor->data + offset, size);
    } નહીંતર {
        apir_buffer_get_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), tensor, data, offset, size);
    }
}

સ્થિર શૂન્ય ggml_backend_remoting_buffer_set_tensor_from_ptr(ggml_backend_buffer_t buffer,
                                                             ggml_tensor *         tensor,
                                                             અચલ શૂન્ય *          data,
                                                             size_t                offset,
                                                             size_t                size) {
    UNUSED(buffer);

    memcpy((અક્ષર *) tensor->data + offset, data, size);

    પાછા;
}

સ્થિર શૂન્ય ggml_backend_remoting_buffer_get_tensor_from_ptr(ggml_backend_buffer_t buffer,
                                                             અચલ ggml_tensor *   tensor,
                                                             શૂન્ય *                data,
                                                             size_t                offset,
                                                             size_t                size) {
    UNUSED(buffer);

    memcpy(data, (અચલ અક્ષર *) tensor->data + offset, size);
}

સ્થિર તાર્કિક ggml_backend_remoting_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                    અચલ ggml_tensor *   src,
                                                    ggml_tensor *         dst) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    તાર્કિક ret = apir_buffer_cpy_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), src, dst);

    પાછા ret;
}

સ્થિર શૂન્ય ggml_backend_remoting_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    apir_buffer_clear(gpu, BUFFER_TO_APIR_CONTEXT(buffer), value);

    પાછા;
}

સ્થિર શૂન્ય ggml_backend_remoting_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    apir_buffer_free_buffer(gpu, BUFFER_TO_APIR_CONTEXT(buffer));

    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    મુક્ત(context);
    buffer->context = ખાલી;
}

અચલ ggml_backend_buffer_i ggml_backend_remoting_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_remoting_buffer_get_base,
    /* .init_tensor     = */ ખાલી,
    /* .memset_tensor   = */ ખાલી,
    /* .set_tensor      = */ ggml_backend_remoting_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_remoting_buffer_get_tensor,
    /* .set_tensor_2d   = */ ખાલી,
    /* .get_tensor_2d   = */ ખાલી,
    /* .cpy_tensor      = */ ggml_backend_remoting_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_buffer_clear,
    /* .reset           = */ ખાલી,
};

અચલ ggml_backend_buffer_i ggml_backend_remoting_buffer_from_ptr_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_remoting_buffer_get_base,
    /* .init_tensor     = */ ખાલી,
    /* .memset_tensor   = */ ખાલી,
    /* .set_tensor      = */ ggml_backend_remoting_buffer_set_tensor_from_ptr,
    /* .get_tensor      = */ ggml_backend_remoting_buffer_get_tensor_from_ptr,
    /* .set_tensor_2d   = */ ખાલી,
    /* .get_tensor_2d   = */ ખાલી,
    /* .cpy_tensor      = */ ggml_backend_remoting_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_buffer_clear,
    /* .reset           = */ ખાલી,
};
