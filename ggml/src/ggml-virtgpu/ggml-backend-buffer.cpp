#شامل "ggml-remoting.h"

#define BUFFER_TO_GPU(name) ((ggml_backend_remoting_buffer_context *) (name)->context)->gpu

ایستا تهی * ggml_backend_remoting_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) buffer->context;
    اگر (context->base) {
        بازگشت context->base;
    }

    context->base = apir_buffer_get_base(BUFFER_TO_GPU(buffer), BUFFER_TO_APIR_CONTEXT(buffer));

    بازگشت context->base;
}

ایستا تهی ggml_backend_remoting_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                    ggml_tensor *         tensor,
                                                    ثابت تهی *          data,
                                                    size_t                offset,
                                                    size_t                size) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    اگر (context->is_from_ptr) {
        memcpy((نویسه *) tensor->data + offset, data, size);
    } وگرنه {
        apir_buffer_set_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), tensor, data, offset, size);
    }

    بازگشت;
}

ایستا تهی ggml_backend_remoting_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                    ثابت ggml_tensor *   tensor,
                                                    تهی *                data,
                                                    size_t                offset,
                                                    size_t                size) {
    virtgpu *                              gpu     = BUFFER_TO_GPU(buffer);
    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    اگر (context->is_from_ptr) {
        memcpy(data, (ثابت نویسه *) tensor->data + offset, size);
    } وگرنه {
        apir_buffer_get_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), tensor, data, offset, size);
    }
}

ایستا تهی ggml_backend_remoting_buffer_set_tensor_from_ptr(ggml_backend_buffer_t buffer,
                                                             ggml_tensor *         tensor,
                                                             ثابت تهی *          data,
                                                             size_t                offset,
                                                             size_t                size) {
    UNUSED(buffer);

    memcpy((نویسه *) tensor->data + offset, data, size);

    بازگشت;
}

ایستا تهی ggml_backend_remoting_buffer_get_tensor_from_ptr(ggml_backend_buffer_t buffer,
                                                             ثابت ggml_tensor *   tensor,
                                                             تهی *                data,
                                                             size_t                offset,
                                                             size_t                size) {
    UNUSED(buffer);

    memcpy(data, (ثابت نویسه *) tensor->data + offset, size);
}

ایستا منطقی ggml_backend_remoting_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                    ثابت ggml_tensor *   src,
                                                    ggml_tensor *         dst) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    منطقی ret = apir_buffer_cpy_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), src, dst);

    بازگشت ret;
}

ایستا تهی ggml_backend_remoting_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    apir_buffer_clear(gpu, BUFFER_TO_APIR_CONTEXT(buffer), value);

    بازگشت;
}

ایستا تهی ggml_backend_remoting_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    apir_buffer_free_buffer(gpu, BUFFER_TO_APIR_CONTEXT(buffer));

    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    آزاد(context);
    buffer->context = پوچ;
}

ثابت ggml_backend_buffer_i ggml_backend_remoting_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_remoting_buffer_get_base,
    /* .init_tensor     = */ پوچ,
    /* .memset_tensor   = */ پوچ,
    /* .set_tensor      = */ ggml_backend_remoting_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_remoting_buffer_get_tensor,
    /* .set_tensor_2d   = */ پوچ,
    /* .get_tensor_2d   = */ پوچ,
    /* .cpy_tensor      = */ ggml_backend_remoting_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_buffer_clear,
    /* .reset           = */ پوچ,
};

ثابت ggml_backend_buffer_i ggml_backend_remoting_buffer_from_ptr_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_remoting_buffer_get_base,
    /* .init_tensor     = */ پوچ,
    /* .memset_tensor   = */ پوچ,
    /* .set_tensor      = */ ggml_backend_remoting_buffer_set_tensor_from_ptr,
    /* .get_tensor      = */ ggml_backend_remoting_buffer_get_tensor_from_ptr,
    /* .set_tensor_2d   = */ پوچ,
    /* .get_tensor_2d   = */ پوچ,
    /* .cpy_tensor      = */ ggml_backend_remoting_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_buffer_clear,
    /* .reset           = */ پوچ,
};
