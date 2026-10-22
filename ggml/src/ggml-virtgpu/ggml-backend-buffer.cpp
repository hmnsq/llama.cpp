#ضمن "ggml-remoting.h"

#تعریف_کړه بفر_منزل_GPU(name) ((ggml_backend_remoting_بفر_context *) (name)->context)->gpu

ساكن فراغ * ggml_backend_remoting_بفر_get_بنسټ(ggml_backend_بفر_t بفر) {
    ggml_backend_remoting_بفر_context * context = (ggml_backend_remoting_بفر_context *) بفر->context;
    إذا (context->بنسټ) {
        أرجع context->بنسټ;
    }

    context->بنسټ = apir_بفر_get_بنسټ(بفر_منزل_GPU(بفر), بفر_منزل_APIR_CONTEXT(بفر));

    أرجع context->بنسټ;
}

ساكن فراغ ggml_backend_remoting_بفر_سېټ_tensor(ggml_backend_بفر_t بفر,
                                                    ggml_tensor *         tensor,
                                                    ثابت فراغ *          معلومات,
                                                    size_t                دننه_واټن,
                                                    size_t                size) {
    virtgpu * gpu = بفر_منزل_GPU(بفر);

    ggml_backend_remoting_بفر_context * context = بفر_منزل_GGML_CONTEXT(بفر);
    إذا (context->is_from_ptr) {
        حافظه_نقل((حرف *) tensor->معلومات + دننه_واټن, معلومات, size);
    } وإلا {
        apir_بفر_سېټ_tensor(gpu, بفر_منزل_APIR_CONTEXT(بفر), tensor, معلومات, دننه_واټن, size);
    }

    أرجع;
}

ساكن فراغ ggml_backend_remoting_بفر_get_tensor(ggml_backend_بفر_t بفر,
                                                    ثابت ggml_tensor *   tensor,
                                                    فراغ *                معلومات,
                                                    size_t                دننه_واټن,
                                                    size_t                size) {
    virtgpu *                              gpu     = بفر_منزل_GPU(بفر);
    ggml_backend_remoting_بفر_context * context = بفر_منزل_GGML_CONTEXT(بفر);
    إذا (context->is_from_ptr) {
        حافظه_نقل(معلومات, (ثابت حرف *) tensor->معلومات + دننه_واټن, size);
    } وإلا {
        apir_بفر_get_tensor(gpu, بفر_منزل_APIR_CONTEXT(بفر), tensor, معلومات, دننه_واټن, size);
    }
}

ساكن فراغ ggml_backend_remoting_بفر_سېټ_tensor_from_ptr(ggml_backend_بفر_t بفر,
                                                             ggml_tensor *         tensor,
                                                             ثابت فراغ *          معلومات,
                                                             size_t                دننه_واټن,
                                                             size_t                size) {
    ناکارول_شوی(بفر);

    حافظه_نقل((حرف *) tensor->معلومات + دننه_واټن, معلومات, size);

    أرجع;
}

ساكن فراغ ggml_backend_remoting_بفر_get_tensor_from_ptr(ggml_backend_بفر_t بفر,
                                                             ثابت ggml_tensor *   tensor,
                                                             فراغ *                معلومات,
                                                             size_t                دننه_واټن,
                                                             size_t                size) {
    ناکارول_شوی(بفر);

    حافظه_نقل(معلومات, (ثابت حرف *) tensor->معلومات + دننه_واټن, size);
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
    /* .get_base        = */ ggml_backend_remoting_بفر_get_بنسټ,
    /* .init_tensor     = */ عدم,
    /* .memset_tensor   = */ عدم,
    /* .set_tensor      = */ ggml_backend_remoting_بفر_سېټ_tensor,
    /* .get_tensor      = */ ggml_backend_remoting_بفر_get_tensor,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_remoting_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_بفر_clear,
    /* .reset           = */ عدم,
};

ثابت ggml_backend_بفر_i ggml_backend_remoting_بفر_from_ptr_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_remoting_بفر_get_بنسټ,
    /* .init_tensor     = */ عدم,
    /* .memset_tensor   = */ عدم,
    /* .set_tensor      = */ ggml_backend_remoting_بفر_سېټ_tensor_from_ptr,
    /* .get_tensor      = */ ggml_backend_remoting_بفر_get_tensor_from_ptr,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_remoting_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_بفر_clear,
    /* .reset           = */ عدم,
};
