#ضمن "ggml-remoting.h"

#وصف_ڪريو بفر_منزل_GPU(name) ((ggml_backend_remoting_بفر_context *) (name)->context)->gpu

ساكن فراغ * ggml_backend_remoting_بفر_get_بنياد(ggml_backend_بفر_t بفر) {
    ggml_backend_remoting_بفر_context * context = (ggml_backend_remoting_بفر_context *) بفر->context;
    إذا (context->بنياد) {
        أرجع context->بنياد;
    }

    context->بنياد = apir_بفر_get_بنياد(بفر_منزل_GPU(بفر), بفر_منزل_APIR_CONTEXT(بفر));

    أرجع context->بنياد;
}

ساكن فراغ ggml_backend_remoting_بفر_سيٽ_tensor(ggml_backend_بفر_t بفر,
                                                    ggml_tensor *         tensor,
                                                    ثابت فراغ *          معلومات,
                                                    size_t                وٿي,
                                                    size_t                size) {
    virtgpu * gpu = بفر_منزل_GPU(بفر);

    ggml_backend_remoting_بفر_context * context = بفر_منزل_GGML_CONTEXT(بفر);
    إذا (context->is_from_ptr) {
        يادگيري_نقل((حرف *) tensor->معلومات + وٿي, معلومات, size);
    } وإلا {
        apir_بفر_سيٽ_tensor(gpu, بفر_منزل_APIR_CONTEXT(بفر), tensor, معلومات, وٿي, size);
    }

    أرجع;
}

ساكن فراغ ggml_backend_remoting_بفر_get_tensor(ggml_backend_بفر_t بفر,
                                                    ثابت ggml_tensor *   tensor,
                                                    فراغ *                معلومات,
                                                    size_t                وٿي,
                                                    size_t                size) {
    virtgpu *                              gpu     = بفر_منزل_GPU(بفر);
    ggml_backend_remoting_بفر_context * context = بفر_منزل_GGML_CONTEXT(بفر);
    إذا (context->is_from_ptr) {
        يادگيري_نقل(معلومات, (ثابت حرف *) tensor->معلومات + وٿي, size);
    } وإلا {
        apir_بفر_get_tensor(gpu, بفر_منزل_APIR_CONTEXT(بفر), tensor, معلومات, وٿي, size);
    }
}

ساكن فراغ ggml_backend_remoting_بفر_سيٽ_tensor_from_ptr(ggml_backend_بفر_t بفر,
                                                             ggml_tensor *         tensor,
                                                             ثابت فراغ *          معلومات,
                                                             size_t                وٿي,
                                                             size_t                size) {
    اڻ_استعمال_ٿيل(بفر);

    يادگيري_نقل((حرف *) tensor->معلومات + وٿي, معلومات, size);

    أرجع;
}

ساكن فراغ ggml_backend_remoting_بفر_get_tensor_from_ptr(ggml_backend_بفر_t بفر,
                                                             ثابت ggml_tensor *   tensor,
                                                             فراغ *                معلومات,
                                                             size_t                وٿي,
                                                             size_t                size) {
    اڻ_استعمال_ٿيل(بفر);

    يادگيري_نقل(معلومات, (ثابت حرف *) tensor->معلومات + وٿي, size);
}

ساكن منطقي ggml_backend_remoting_بفر_cpy_tensor(ggml_backend_بفر_t بفر,
                                                    ثابت ggml_tensor *   src,
                                                    ggml_tensor *         dst) {
    virtgpu * gpu = بفر_منزل_GPU(بفر);

    منطقي ret = apir_بفر_cpy_tensor(gpu, بفر_منزل_APIR_CONTEXT(بفر), src, dst);

    أرجع ret;
}

ساكن فراغ ggml_backend_remoting_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    virtgpu * gpu = بفر_منزل_GPU(بفر);

    apir_بفر_clear(gpu, بفر_منزل_APIR_CONTEXT(بفر), value);

    أرجع;
}

ساكن فراغ ggml_backend_remoting_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    virtgpu * gpu = بفر_منزل_GPU(بفر);

    apir_بفر_free_بفر(gpu, بفر_منزل_APIR_CONTEXT(بفر));

    ggml_backend_remoting_بفر_context * context = بفر_منزل_GGML_CONTEXT(بفر);
    حرر(context);
    بفر->context = عدم;
}

ثابت ggml_backend_بفر_i ggml_backend_remoting_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_remoting_بفر_get_بنياد,
    /* .init_tensor     = */ عدم,
    /* .memset_tensor   = */ عدم,
    /* .set_tensor      = */ ggml_backend_remoting_بفر_سيٽ_tensor,
    /* .get_tensor      = */ ggml_backend_remoting_بفر_get_tensor,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_remoting_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_بفر_clear,
    /* .reset           = */ عدم,
};

ثابت ggml_backend_بفر_i ggml_backend_remoting_بفر_from_ptr_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_remoting_بفر_get_بنياد,
    /* .init_tensor     = */ عدم,
    /* .memset_tensor   = */ عدم,
    /* .set_tensor      = */ ggml_backend_remoting_بفر_سيٽ_tensor_from_ptr,
    /* .get_tensor      = */ ggml_backend_remoting_بفر_get_tensor_from_ptr,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_remoting_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_بفر_clear,
    /* .reset           = */ عدم,
};
