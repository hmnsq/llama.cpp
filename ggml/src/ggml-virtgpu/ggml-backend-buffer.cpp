#अन्तर्भाव "ggml-remoting.h"

#define BUFFER_TO_GPU(name) ((ggml_backend_remoting_buffer_context *) (name)->context)->gpu

स्थिर शून्य * ggml_backend_remoting_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_remoting_buffer_context * context = (ggml_backend_remoting_buffer_context *) buffer->context;
    यदि (context->base) {
        प्रत्यागम context->base;
    }

    context->base = apir_buffer_get_base(BUFFER_TO_GPU(buffer), BUFFER_TO_APIR_CONTEXT(buffer));

    प्रत्यागम context->base;
}

स्थिर शून्य ggml_backend_remoting_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                    ggml_tensor *         tensor,
                                                    ध्रुव शून्य *          data,
                                                    माप_प्रकार                offset,
                                                    माप_प्रकार                size) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    यदि (context->is_from_ptr) {
        स_नकल((वर्ण *) tensor->data + offset, data, size);
    } अन्यथा {
        apir_buffer_set_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), tensor, data, offset, size);
    }

    प्रत्यागम;
}

स्थिर शून्य ggml_backend_remoting_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                    ध्रुव ggml_tensor *   tensor,
                                                    शून्य *                data,
                                                    माप_प्रकार                offset,
                                                    माप_प्रकार                size) {
    virtgpu *                              gpu     = BUFFER_TO_GPU(buffer);
    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    यदि (context->is_from_ptr) {
        स_नकल(data, (ध्रुव वर्ण *) tensor->data + offset, size);
    } अन्यथा {
        apir_buffer_get_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), tensor, data, offset, size);
    }
}

स्थिर शून्य ggml_backend_remoting_buffer_set_tensor_from_ptr(ggml_backend_buffer_t buffer,
                                                             ggml_tensor *         tensor,
                                                             ध्रुव शून्य *          data,
                                                             माप_प्रकार                offset,
                                                             माप_प्रकार                size) {
    UNUSED(buffer);

    स_नकल((वर्ण *) tensor->data + offset, data, size);

    प्रत्यागम;
}

स्थिर शून्य ggml_backend_remoting_buffer_get_tensor_from_ptr(ggml_backend_buffer_t buffer,
                                                             ध्रुव ggml_tensor *   tensor,
                                                             शून्य *                data,
                                                             माप_प्रकार                offset,
                                                             माप_प्रकार                size) {
    UNUSED(buffer);

    स_नकल(data, (ध्रुव वर्ण *) tensor->data + offset, size);
}

स्थिर तार्किक ggml_backend_remoting_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                    ध्रुव ggml_tensor *   src,
                                                    ggml_tensor *         dst) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    तार्किक ret = apir_buffer_cpy_tensor(gpu, BUFFER_TO_APIR_CONTEXT(buffer), src, dst);

    प्रत्यागम ret;
}

स्थिर शून्य ggml_backend_remoting_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    apir_buffer_clear(gpu, BUFFER_TO_APIR_CONTEXT(buffer), value);

    प्रत्यागम;
}

स्थिर शून्य ggml_backend_remoting_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    virtgpu * gpu = BUFFER_TO_GPU(buffer);

    apir_buffer_free_buffer(gpu, BUFFER_TO_APIR_CONTEXT(buffer));

    ggml_backend_remoting_buffer_context * context = BUFFER_TO_GGML_CONTEXT(buffer);
    मुक्त(context);
    buffer->context = रिक्त;
}

ध्रुव ggml_backend_buffer_i ggml_backend_remoting_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_remoting_buffer_get_base,
    /* .init_tensor     = */ रिक्त,
    /* .memset_tensor   = */ रिक्त,
    /* .set_tensor      = */ ggml_backend_remoting_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_remoting_buffer_get_tensor,
    /* .set_tensor_2d   = */ रिक्त,
    /* .get_tensor_2d   = */ रिक्त,
    /* .cpy_tensor      = */ ggml_backend_remoting_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_buffer_clear,
    /* .reset           = */ रिक्त,
};

ध्रुव ggml_backend_buffer_i ggml_backend_remoting_buffer_from_ptr_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_remoting_buffer_get_base,
    /* .init_tensor     = */ रिक्त,
    /* .memset_tensor   = */ रिक्त,
    /* .set_tensor      = */ ggml_backend_remoting_buffer_set_tensor_from_ptr,
    /* .get_tensor      = */ ggml_backend_remoting_buffer_get_tensor_from_ptr,
    /* .set_tensor_2d   = */ रिक्त,
    /* .get_tensor_2d   = */ रिक्त,
    /* .cpy_tensor      = */ ggml_backend_remoting_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_buffer_clear,
    /* .reset           = */ रिक्त,
};
