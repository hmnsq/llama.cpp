#شامل "ggml-remoting.h"

#تعریف_کرو بفر_مقصد_GPU(name) ((ggml_backend_remoting_بفر_context *) (name)->context)->gpu

ساکن خالی * ggml_backend_remoting_بفر_get_بنیاد(ggml_backend_بفر_t بفر) {
    ggml_backend_remoting_بفر_context * context = (ggml_backend_remoting_بفر_context *) بفر->context;
    اگر (context->بنیاد) {
        واپس context->بنیاد;
    }

    context->بنیاد = apir_بفر_get_بنیاد(بفر_مقصد_GPU(بفر), بفر_مقصد_APIR_CONTEXT(بفر));

    واپس context->بنیاد;
}

ساکن خالی ggml_backend_remoting_بفر_سیٹ_tensor(ggml_backend_بفر_t بفر,
                                                    ggml_tensor *         tensor,
                                                    ثابت خالی *          ڈیٹا,
                                                    size_t                انحراف,
                                                    size_t                size) {
    virtgpu * gpu = بفر_مقصد_GPU(بفر);

    ggml_backend_remoting_بفر_context * context = بفر_مقصد_GGML_CONTEXT(بفر);
    اگر (context->is_from_ptr) {
        حافظہ_نقل((حرف *) tensor->ڈیٹا + انحراف, ڈیٹا, size);
    } ورنہ {
        apir_بفر_سیٹ_tensor(gpu, بفر_مقصد_APIR_CONTEXT(بفر), tensor, ڈیٹا, انحراف, size);
    }

    واپس;
}

ساکن خالی ggml_backend_remoting_بفر_get_tensor(ggml_backend_بفر_t بفر,
                                                    ثابت ggml_tensor *   tensor,
                                                    خالی *                ڈیٹا,
                                                    size_t                انحراف,
                                                    size_t                size) {
    virtgpu *                              gpu     = بفر_مقصد_GPU(بفر);
    ggml_backend_remoting_بفر_context * context = بفر_مقصد_GGML_CONTEXT(بفر);
    اگر (context->is_from_ptr) {
        حافظہ_نقل(ڈیٹا, (ثابت حرف *) tensor->ڈیٹا + انحراف, size);
    } ورنہ {
        apir_بفر_get_tensor(gpu, بفر_مقصد_APIR_CONTEXT(بفر), tensor, ڈیٹا, انحراف, size);
    }
}

ساکن خالی ggml_backend_remoting_بفر_سیٹ_tensor_from_ptr(ggml_backend_بفر_t بفر,
                                                             ggml_tensor *         tensor,
                                                             ثابت خالی *          ڈیٹا,
                                                             size_t                انحراف,
                                                             size_t                size) {
    غیر_مستعمل(بفر);

    حافظہ_نقل((حرف *) tensor->ڈیٹا + انحراف, ڈیٹا, size);

    واپس;
}

ساکن خالی ggml_backend_remoting_بفر_get_tensor_from_ptr(ggml_backend_بفر_t بفر,
                                                             ثابت ggml_tensor *   tensor,
                                                             خالی *                ڈیٹا,
                                                             size_t                انحراف,
                                                             size_t                size) {
    غیر_مستعمل(بفر);

    حافظہ_نقل(ڈیٹا, (ثابت حرف *) tensor->ڈیٹا + انحراف, size);
}

ساکن منطقی ggml_backend_remoting_بفر_cpy_tensor(ggml_backend_بفر_t بفر,
                                                    ثابت ggml_tensor *   src,
                                                    ggml_tensor *         dst) {
    virtgpu * gpu = بفر_مقصد_GPU(بفر);

    منطقی ret = apir_بفر_cpy_tensor(gpu, بفر_مقصد_APIR_CONTEXT(بفر), src, dst);

    واپس ret;
}

ساکن خالی ggml_backend_remoting_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    virtgpu * gpu = بفر_مقصد_GPU(بفر);

    apir_بفر_clear(gpu, بفر_مقصد_APIR_CONTEXT(بفر), value);

    واپس;
}

ساکن خالی ggml_backend_remoting_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    virtgpu * gpu = بفر_مقصد_GPU(بفر);

    apir_بفر_free_بفر(gpu, بفر_مقصد_APIR_CONTEXT(بفر));

    ggml_backend_remoting_بفر_context * context = بفر_مقصد_GGML_CONTEXT(بفر);
    آزاد(context);
    بفر->context = تہی;
}

ثابت ggml_backend_بفر_i ggml_backend_remoting_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_remoting_بفر_get_بنیاد,
    /* .init_tensor     = */ تہی,
    /* .memset_tensor   = */ تہی,
    /* .set_tensor      = */ ggml_backend_remoting_بفر_سیٹ_tensor,
    /* .get_tensor      = */ ggml_backend_remoting_بفر_get_tensor,
    /* .set_tensor_2d   = */ تہی,
    /* .get_tensor_2d   = */ تہی,
    /* .cpy_tensor      = */ ggml_backend_remoting_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_بفر_clear,
    /* .reset           = */ تہی,
};

ثابت ggml_backend_بفر_i ggml_backend_remoting_بفر_from_ptr_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_remoting_بفر_get_بنیاد,
    /* .init_tensor     = */ تہی,
    /* .memset_tensor   = */ تہی,
    /* .set_tensor      = */ ggml_backend_remoting_بفر_سیٹ_tensor_from_ptr,
    /* .get_tensor      = */ ggml_backend_remoting_بفر_get_tensor_from_ptr,
    /* .set_tensor_2d   = */ تہی,
    /* .get_tensor_2d   = */ تہی,
    /* .cpy_tensor      = */ ggml_backend_remoting_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_بفر_clear,
    /* .reset           = */ تہی,
};
