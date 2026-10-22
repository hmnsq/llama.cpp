#अन्तर्भाव "ggml-remoting.h"

#परिभाषय बफर_गन्तव्यम्_GPU(name) ((ggml_backend_remoting_बफर_context *) (name)->context)->gpu

स्थिर शून्य * ggml_backend_remoting_बफर_get_आधारः(ggml_backend_बफर_t बफर) {
    ggml_backend_remoting_बफर_context * context = (ggml_backend_remoting_बफर_context *) बफर->context;
    यदि (context->आधारः) {
        प्रत्यागम context->आधारः;
    }

    context->आधारः = apir_बफर_get_आधारः(बफर_गन्तव्यम्_GPU(बफर), बफर_गन्तव्यम्_APIR_CONTEXT(बफर));

    प्रत्यागम context->आधारः;
}

स्थिर शून्य ggml_backend_remoting_बफर_समुच्चयः_tensor(ggml_backend_बफर_t बफर,
                                                    ggml_tensor *         tensor,
                                                    ध्रुव शून्य *          दत्त,
                                                    size_t                विस्थापनम्,
                                                    size_t                size) {
    virtgpu * gpu = बफर_गन्तव्यम्_GPU(बफर);

    ggml_backend_remoting_बफर_context * context = बफर_गन्तव्यम्_GGML_CONTEXT(बफर);
    यदि (context->is_from_ptr) {
        स्मृतिं_प्रतिलिख((वर्ण *) tensor->दत्त + विस्थापनम्, दत्त, size);
    } अन्यथा {
        apir_बफर_समुच्चयः_tensor(gpu, बफर_गन्तव्यम्_APIR_CONTEXT(बफर), tensor, दत्त, विस्थापनम्, size);
    }

    प्रत्यागम;
}

स्थिर शून्य ggml_backend_remoting_बफर_get_tensor(ggml_backend_बफर_t बफर,
                                                    ध्रुव ggml_tensor *   tensor,
                                                    शून्य *                दत्त,
                                                    size_t                विस्थापनम्,
                                                    size_t                size) {
    virtgpu *                              gpu     = बफर_गन्तव्यम्_GPU(बफर);
    ggml_backend_remoting_बफर_context * context = बफर_गन्तव्यम्_GGML_CONTEXT(बफर);
    यदि (context->is_from_ptr) {
        स्मृतिं_प्रतिलिख(दत्त, (ध्रुव वर्ण *) tensor->दत्त + विस्थापनम्, size);
    } अन्यथा {
        apir_बफर_get_tensor(gpu, बफर_गन्तव्यम्_APIR_CONTEXT(बफर), tensor, दत्त, विस्थापनम्, size);
    }
}

स्थिर शून्य ggml_backend_remoting_बफर_समुच्चयः_tensor_from_ptr(ggml_backend_बफर_t बफर,
                                                             ggml_tensor *         tensor,
                                                             ध्रुव शून्य *          दत्त,
                                                             size_t                विस्थापनम्,
                                                             size_t                size) {
    अप्रयुक्तम्(बफर);

    स्मृतिं_प्रतिलिख((वर्ण *) tensor->दत्त + विस्थापनम्, दत्त, size);

    प्रत्यागम;
}

स्थिर शून्य ggml_backend_remoting_बफर_get_tensor_from_ptr(ggml_backend_बफर_t बफर,
                                                             ध्रुव ggml_tensor *   tensor,
                                                             शून्य *                दत्त,
                                                             size_t                विस्थापनम्,
                                                             size_t                size) {
    अप्रयुक्तम्(बफर);

    स्मृतिं_प्रतिलिख(दत्त, (ध्रुव वर्ण *) tensor->दत्त + विस्थापनम्, size);
}

स्थिर तार्किक ggml_backend_remoting_बफर_cpy_tensor(ggml_backend_बफर_t बफर,
                                                    ध्रुव ggml_tensor *   src,
                                                    ggml_tensor *         dst) {
    virtgpu * gpu = बफर_गन्तव्यम्_GPU(बफर);

    तार्किक ret = apir_बफर_cpy_tensor(gpu, बफर_गन्तव्यम्_APIR_CONTEXT(बफर), src, dst);

    प्रत्यागम ret;
}

स्थिर शून्य ggml_backend_remoting_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    virtgpu * gpu = बफर_गन्तव्यम्_GPU(बफर);

    apir_बफर_clear(gpu, बफर_गन्तव्यम्_APIR_CONTEXT(बफर), value);

    प्रत्यागम;
}

स्थिर शून्य ggml_backend_remoting_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    virtgpu * gpu = बफर_गन्तव्यम्_GPU(बफर);

    apir_बफर_free_बफर(gpu, बफर_गन्तव्यम्_APIR_CONTEXT(बफर));

    ggml_backend_remoting_बफर_context * context = बफर_गन्तव्यम्_GGML_CONTEXT(बफर);
    मुक्त(context);
    बफर->context = रिक्त;
}

ध्रुव ggml_backend_बफर_i ggml_backend_remoting_बफर_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_remoting_बफर_get_आधारः,
    /* .init_tensor     = */ रिक्त,
    /* .memset_tensor   = */ रिक्त,
    /* .set_tensor      = */ ggml_backend_remoting_बफर_समुच्चयः_tensor,
    /* .get_tensor      = */ ggml_backend_remoting_बफर_get_tensor,
    /* .set_tensor_2d   = */ रिक्त,
    /* .get_tensor_2d   = */ रिक्त,
    /* .cpy_tensor      = */ ggml_backend_remoting_बफर_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_बफर_clear,
    /* .reset           = */ रिक्त,
};

ध्रुव ggml_backend_बफर_i ggml_backend_remoting_बफर_from_ptr_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_remoting_बफर_get_आधारः,
    /* .init_tensor     = */ रिक्त,
    /* .memset_tensor   = */ रिक्त,
    /* .set_tensor      = */ ggml_backend_remoting_बफर_समुच्चयः_tensor_from_ptr,
    /* .get_tensor      = */ ggml_backend_remoting_बफर_get_tensor_from_ptr,
    /* .set_tensor_2d   = */ रिक्त,
    /* .get_tensor_2d   = */ रिक्त,
    /* .cpy_tensor      = */ ggml_backend_remoting_बफर_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_बफर_clear,
    /* .reset           = */ रिक्त,
};
