#ضمن "ggml-remoting.h"

#define BUFFER_TO_GPU(name) ((ggml_backend_remoting_buffer_context *) (name)->context)->gpu

ساكن فراغ * ggml_backend_remoting_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) buffer->context;
    إذا (context->base) {
        إرجاع context->base;
    }

    context->base = apir_buffer_get_base(BUFFER_TO_GPU(buffer), BUFFER_TO_APIR_CONTEXT(buffer));

    إرجاع context->base;
}

ساكن فراغ ggml_backend_remoting_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                    ggml_tensor *         tensor,
                                                    ثابت فراغ *          data,
                                                    size_t                offset,
                                                    size_t                size) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    إذا (context->is_from_ptr) {
        memcpy((حرف *) tensor->data + offset, data, size);
    } وإلا {
        apir_buffer_set_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), tensor, data, offset, size);
    }

    إرجاع;
}

ساكن فراغ ggml_backend_remoting_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                    ثابت ggml_tensor *   tensor,
                                                    فراغ *                data,
                                                    size_t                offset,
                                                    size_t                size) {
    virtgpu *                              gpu     = BUFFER_TO_GPU(buffer);
    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    إذا (context->is_from_ptr) {
        memcpy(data, (ثابت حرف *) tensor->data + offset, size);
    } وإلا {
        apir_buffer_get_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), tensor, data, offset, size);
    }
}

ساكن فراغ ggml_backend_remoting_buffer_set_tensor_from_ptr(ggml_backend_buffer_t buffer,
                                                             ggml_tensor *         tensor,
                                                             ثابت فراغ *          data,
                                                             size_t                offset,
                                                             size_t                size) {
    UNUSED(buffer);

    memcpy((حرف *) tensor->data + offset, data, size);

    إرجاع;
}

ساكن فراغ ggml_backend_remoting_buffer_get_tensor_from_ptr(ggml_backend_buffer_t buffer,
                                                             ثابت ggml_tensor *   tensor,
                                                             فراغ *                data,
                                                             size_t                offset,
                                                             size_t                size) {
    UNUSED(buffer);

    memcpy(data, (ثابت حرف *) tensor->data + offset, size);
}

ساكن منطقي ggml_backend_remoting_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                    ثابت ggml_tensor *   src,
                                                    ggml_tensor *         dst) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    منطقي ret = apir_buffer_cpy_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), src, dst);

    إرجاع ret;
}

ساكن فراغ ggml_backend_remoting_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    apir_buffer_clear(gpu, BUFFER_TO_APIR_CONTEXT(buffer), value);

    إرجاع;
}

ساكن فراغ ggml_backend_remoting_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    apir_buffer_free_buffer(gpu, BUFFER_TO_APIR_CONTEXT(buffer));

    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    حرر(context);
    buffer->context = عدم;
}

ثابت ggml_backend_buffer_i ggml_backend_remoting_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_remoting_buffer_get_base,
    /* .init_tensor     = */ عدم,
    /* .memset_tensor   = */ عدم,
    /* .set_tensor      = */ ggml_backend_remoting_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_remoting_buffer_get_tensor,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_remoting_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_buffer_clear,
    /* .reset           = */ عدم,
};

ثابت ggml_backend_buffer_i ggml_backend_remoting_buffer_from_ptr_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_remoting_buffer_get_base,
    /* .init_tensor     = */ عدم,
    /* .memset_tensor   = */ عدم,
    /* .set_tensor      = */ ggml_backend_remoting_buffer_set_tensor_from_ptr,
    /* .get_tensor      = */ ggml_backend_remoting_buffer_get_tensor_from_ptr,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_remoting_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_buffer_clear,
    /* .reset           = */ عدم,
};
