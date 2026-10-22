#समावेश "ggml-remoting.h"

#घोषणा बफर_TO_GPU(name) ((ggml_backend_remoting_बफर_context *) (name)->context)->gpu

अटल व्योम * ggml_backend_remoting_बफर_get_base(ggml_backend_बफर_t बफर) {
    ggml_backend_remoting_बफर_context * context = (ggml_backend_remoting_बफर_context *) बफर->context;
    अगर (context->base) {
        वापस context->base;
    }

    context->base = apir_बफर_get_base(बफर_TO_GPU(बफर), बफर_TO_APIR_CONTEXT(बफर));

    वापस context->base;
}

अटल व्योम ggml_backend_remoting_बफर_set_tensor(ggml_backend_बफर_t बफर,
                                                    ggml_tensor *         tensor,
                                                    स्थिर व्योम *          तथ्य,
                                                    माप_प्रकार                offset,
                                                    माप_प्रकार                size) {
    virtgpu * gpu = बफर_TO_GPU(बफर);

    ggml_backend_remoting_बफर_context * context = बफर_TO_GGML_CONTEXT(बफर);
    अगर (context->is_from_ptr) {
        स_नकल((अक्षर *) tensor->तथ्य + offset, तथ्य, size);
    } अन्यथा {
        apir_बफर_set_tensor(gpu, बफर_TO_APIR_CONTEXT(बफर), tensor, तथ्य, offset, size);
    }

    वापस;
}

अटल व्योम ggml_backend_remoting_बफर_get_tensor(ggml_backend_बफर_t बफर,
                                                    स्थिर ggml_tensor *   tensor,
                                                    व्योम *                तथ्य,
                                                    माप_प्रकार                offset,
                                                    माप_प्रकार                size) {
    virtgpu *                              gpu     = बफर_TO_GPU(बफर);
    ggml_backend_remoting_बफर_context * context = बफर_TO_GGML_CONTEXT(बफर);
    अगर (context->is_from_ptr) {
        स_नकल(तथ्य, (स्थिर अक्षर *) tensor->तथ्य + offset, size);
    } अन्यथा {
        apir_बफर_get_tensor(gpu, बफर_TO_APIR_CONTEXT(बफर), tensor, तथ्य, offset, size);
    }
}

अटल व्योम ggml_backend_remoting_बफर_set_tensor_from_ptr(ggml_backend_बफर_t बफर,
                                                             ggml_tensor *         tensor,
                                                             स्थिर व्योम *          तथ्य,
                                                             माप_प्रकार                offset,
                                                             माप_प्रकार                size) {
    UNUSED(बफर);

    स_नकल((अक्षर *) tensor->तथ्य + offset, तथ्य, size);

    वापस;
}

अटल व्योम ggml_backend_remoting_बफर_get_tensor_from_ptr(ggml_backend_बफर_t बफर,
                                                             स्थिर ggml_tensor *   tensor,
                                                             व्योम *                तथ्य,
                                                             माप_प्रकार                offset,
                                                             माप_प्रकार                size) {
    UNUSED(बफर);

    स_नकल(तथ्य, (स्थिर अक्षर *) tensor->तथ्य + offset, size);
}

अटल bool ggml_backend_remoting_बफर_cpy_tensor(ggml_backend_बफर_t बफर,
                                                    स्थिर ggml_tensor *   src,
                                                    ggml_tensor *         dst) {
    virtgpu * gpu = बफर_TO_GPU(बफर);

    bool ret = apir_बफर_cpy_tensor(gpu, बफर_TO_APIR_CONTEXT(बफर), src, dst);

    वापस ret;
}

अटल व्योम ggml_backend_remoting_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    virtgpu * gpu = बफर_TO_GPU(बफर);

    apir_बफर_clear(gpu, बफर_TO_APIR_CONTEXT(बफर), value);

    वापस;
}

अटल व्योम ggml_backend_remoting_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    virtgpu * gpu = बफर_TO_GPU(बफर);

    apir_बफर_free_बफर(gpu, बफर_TO_APIR_CONTEXT(बफर));

    ggml_backend_remoting_बफर_context * context = बफर_TO_GGML_CONTEXT(बफर);
    मुक्त(context);
    बफर->context = शून्य;
}

स्थिर ggml_backend_बफर_i ggml_backend_remoting_बफर_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_remoting_बफर_get_base,
    /* .init_tensor     = */ शून्य,
    /* .memset_tensor   = */ शून्य,
    /* .set_tensor      = */ ggml_backend_remoting_बफर_set_tensor,
    /* .get_tensor      = */ ggml_backend_remoting_बफर_get_tensor,
    /* .set_tensor_2d   = */ शून्य,
    /* .get_tensor_2d   = */ शून्य,
    /* .cpy_tensor      = */ ggml_backend_remoting_बफर_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_बफर_clear,
    /* .reset           = */ शून्य,
};

स्थिर ggml_backend_बफर_i ggml_backend_remoting_बफर_from_ptr_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_remoting_बफर_get_base,
    /* .init_tensor     = */ शून्य,
    /* .memset_tensor   = */ शून्य,
    /* .set_tensor      = */ ggml_backend_remoting_बफर_set_tensor_from_ptr,
    /* .get_tensor      = */ ggml_backend_remoting_बफर_get_tensor_from_ptr,
    /* .set_tensor_2d   = */ शून्य,
    /* .get_tensor_2d   = */ शून्य,
    /* .cpy_tensor      = */ ggml_backend_remoting_बफर_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_बफर_clear,
    /* .reset           = */ शून्य,
};
