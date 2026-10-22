#அந்தர்பாவ "ggml-remoting.h"

#define BUFFER_TO_GPU(name) ((ggml_backend_remoting_buffer_context *) (name)->context)->gpu

நிலையான வெறுமை * ggml_backend_remoting_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) buffer->context;
    எனில் (context->base) {
        திரும்பு context->base;
    }

    context->base = apir_buffer_get_base(BUFFER_TO_GPU(buffer), BUFFER_TO_APIR_CONTEXT(buffer));

    திரும்பு context->base;
}

நிலையான வெறுமை ggml_backend_remoting_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                    ggml_tensor *         tensor,
                                                    மாறிலி வெறுமை *          data,
                                                    size_t                offset,
                                                    size_t                size) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    எனில் (context->is_from_ptr) {
        memcpy((எழுத்து *) tensor->data + offset, data, size);
    } இல்லையெனில் {
        apir_buffer_set_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), tensor, data, offset, size);
    }

    திரும்பு;
}

நிலையான வெறுமை ggml_backend_remoting_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                    மாறிலி ggml_tensor *   tensor,
                                                    வெறுமை *                data,
                                                    size_t                offset,
                                                    size_t                size) {
    virtgpu *                              gpu     = BUFFER_TO_GPU(buffer);
    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    எனில் (context->is_from_ptr) {
        memcpy(data, (மாறிலி எழுத்து *) tensor->data + offset, size);
    } இல்லையெனில் {
        apir_buffer_get_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), tensor, data, offset, size);
    }
}

நிலையான வெறுமை ggml_backend_remoting_buffer_set_tensor_from_ptr(ggml_backend_buffer_t buffer,
                                                             ggml_tensor *         tensor,
                                                             மாறிலி வெறுமை *          data,
                                                             size_t                offset,
                                                             size_t                size) {
    UNUSED(buffer);

    memcpy((எழுத்து *) tensor->data + offset, data, size);

    திரும்பு;
}

நிலையான வெறுமை ggml_backend_remoting_buffer_get_tensor_from_ptr(ggml_backend_buffer_t buffer,
                                                             மாறிலி ggml_tensor *   tensor,
                                                             வெறுமை *                data,
                                                             size_t                offset,
                                                             size_t                size) {
    UNUSED(buffer);

    memcpy(data, (மாறிலி எழுத்து *) tensor->data + offset, size);
}

நிலையான தருக்க ggml_backend_remoting_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                    மாறிலி ggml_tensor *   src,
                                                    ggml_tensor *         dst) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    தருக்க ret = apir_buffer_cpy_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), src, dst);

    திரும்பு ret;
}

நிலையான வெறுமை ggml_backend_remoting_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    apir_buffer_clear(gpu, BUFFER_TO_APIR_CONTEXT(buffer), value);

    திரும்பு;
}

நிலையான வெறுமை ggml_backend_remoting_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    apir_buffer_free_buffer(gpu, BUFFER_TO_APIR_CONTEXT(buffer));

    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    விடுவி(context);
    buffer->context = வெற்று;
}

மாறிலி ggml_backend_buffer_i ggml_backend_remoting_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_remoting_buffer_get_base,
    /* .init_tensor     = */ வெற்று,
    /* .memset_tensor   = */ வெற்று,
    /* .set_tensor      = */ ggml_backend_remoting_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_remoting_buffer_get_tensor,
    /* .set_tensor_2d   = */ வெற்று,
    /* .get_tensor_2d   = */ வெற்று,
    /* .cpy_tensor      = */ ggml_backend_remoting_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_buffer_clear,
    /* .reset           = */ வெற்று,
};

மாறிலி ggml_backend_buffer_i ggml_backend_remoting_buffer_from_ptr_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_remoting_buffer_get_base,
    /* .init_tensor     = */ வெற்று,
    /* .memset_tensor   = */ வெற்று,
    /* .set_tensor      = */ ggml_backend_remoting_buffer_set_tensor_from_ptr,
    /* .get_tensor      = */ ggml_backend_remoting_buffer_get_tensor_from_ptr,
    /* .set_tensor_2d   = */ வெற்று,
    /* .get_tensor_2d   = */ வெற்று,
    /* .cpy_tensor      = */ ggml_backend_remoting_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_buffer_clear,
    /* .reset           = */ வெற்று,
};
