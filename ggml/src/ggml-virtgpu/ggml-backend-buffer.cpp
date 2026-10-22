#ضمن "ggml-remoting.h"

#تعریف_کن میانگیر_مقصد_GPU(name) ((ggml_backend_remoting_میانگیر_context *) (name)->context)->gpu

ساكن فراغ * ggml_backend_remoting_میانگیر_get_پایه(ggml_backend_میانگیر_t میانگیر) {
    ggml_backend_remoting_میانگیر_context * context = (ggml_backend_remoting_میانگیر_context *) میانگیر->context;
    إذا (context->پایه) {
        أرجع context->پایه;
    }

    context->پایه = apir_میانگیر_get_پایه(میانگیر_مقصد_GPU(میانگیر), میانگیر_مقصد_APIR_CONTEXT(میانگیر));

    أرجع context->پایه;
}

ساكن فراغ ggml_backend_remoting_میانگیر_مجموعه_tensor(ggml_backend_میانگیر_t میانگیر,
                                                    ggml_tensor *         tensor,
                                                    ثابت فراغ *          داده,
                                                    size_t                جابه_جایی,
                                                    size_t                size) {
    virtgpu * gpu = میانگیر_مقصد_GPU(میانگیر);

    ggml_backend_remoting_میانگیر_context * context = میانگیر_مقصد_GGML_CONTEXT(میانگیر);
    إذا (context->is_from_ptr) {
        کپی_حافظه((حرف *) tensor->داده + جابه_جایی, داده, size);
    } وإلا {
        apir_میانگیر_مجموعه_tensor(gpu, میانگیر_مقصد_APIR_CONTEXT(میانگیر), tensor, داده, جابه_جایی, size);
    }

    أرجع;
}

ساكن فراغ ggml_backend_remoting_میانگیر_get_tensor(ggml_backend_میانگیر_t میانگیر,
                                                    ثابت ggml_tensor *   tensor,
                                                    فراغ *                داده,
                                                    size_t                جابه_جایی,
                                                    size_t                size) {
    virtgpu *                              gpu     = میانگیر_مقصد_GPU(میانگیر);
    ggml_backend_remoting_میانگیر_context * context = میانگیر_مقصد_GGML_CONTEXT(میانگیر);
    إذا (context->is_from_ptr) {
        کپی_حافظه(داده, (ثابت حرف *) tensor->داده + جابه_جایی, size);
    } وإلا {
        apir_میانگیر_get_tensor(gpu, میانگیر_مقصد_APIR_CONTEXT(میانگیر), tensor, داده, جابه_جایی, size);
    }
}

ساكن فراغ ggml_backend_remoting_میانگیر_مجموعه_tensor_from_ptr(ggml_backend_میانگیر_t میانگیر,
                                                             ggml_tensor *         tensor,
                                                             ثابت فراغ *          داده,
                                                             size_t                جابه_جایی,
                                                             size_t                size) {
    استفادهنشده(میانگیر);

    کپی_حافظه((حرف *) tensor->داده + جابه_جایی, داده, size);

    أرجع;
}

ساكن فراغ ggml_backend_remoting_میانگیر_get_tensor_from_ptr(ggml_backend_میانگیر_t میانگیر,
                                                             ثابت ggml_tensor *   tensor,
                                                             فراغ *                داده,
                                                             size_t                جابه_جایی,
                                                             size_t                size) {
    استفادهنشده(میانگیر);

    کپی_حافظه(داده, (ثابت حرف *) tensor->داده + جابه_جایی, size);
}

ساكن منطقي ggml_backend_remoting_میانگیر_cpy_tensor(ggml_backend_میانگیر_t میانگیر,
                                                    ثابت ggml_tensor *   src,
                                                    ggml_tensor *         dst) {
    virtgpu * gpu = میانگیر_مقصد_GPU(میانگیر);

    منطقي ret = apir_میانگیر_cpy_tensor(gpu, میانگیر_مقصد_APIR_CONTEXT(میانگیر), src, dst);

    أرجع ret;
}

ساكن فراغ ggml_backend_remoting_میانگیر_clear(ggml_backend_میانگیر_t میانگیر, uint8_t value) {
    virtgpu * gpu = میانگیر_مقصد_GPU(میانگیر);

    apir_میانگیر_clear(gpu, میانگیر_مقصد_APIR_CONTEXT(میانگیر), value);

    أرجع;
}

ساكن فراغ ggml_backend_remoting_میانگیر_free_میانگیر(ggml_backend_میانگیر_t میانگیر) {
    virtgpu * gpu = میانگیر_مقصد_GPU(میانگیر);

    apir_میانگیر_free_میانگیر(gpu, میانگیر_مقصد_APIR_CONTEXT(میانگیر));

    ggml_backend_remoting_میانگیر_context * context = میانگیر_مقصد_GGML_CONTEXT(میانگیر);
    حرر(context);
    میانگیر->context = عدم;
}

ثابت ggml_backend_میانگیر_i ggml_backend_remoting_میانگیر_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_میانگیر_free_میانگیر,
    /* .get_base        = */ ggml_backend_remoting_میانگیر_get_پایه,
    /* .init_tensor     = */ عدم,
    /* .memset_tensor   = */ عدم,
    /* .set_tensor      = */ ggml_backend_remoting_میانگیر_مجموعه_tensor,
    /* .get_tensor      = */ ggml_backend_remoting_میانگیر_get_tensor,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_remoting_میانگیر_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_میانگیر_clear,
    /* .reset           = */ عدم,
};

ثابت ggml_backend_میانگیر_i ggml_backend_remoting_میانگیر_from_ptr_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_میانگیر_free_میانگیر,
    /* .get_base        = */ ggml_backend_remoting_میانگیر_get_پایه,
    /* .init_tensor     = */ عدم,
    /* .memset_tensor   = */ عدم,
    /* .set_tensor      = */ ggml_backend_remoting_میانگیر_مجموعه_tensor_from_ptr,
    /* .get_tensor      = */ ggml_backend_remoting_میانگیر_get_tensor_from_ptr,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_remoting_میانگیر_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_میانگیر_clear,
    /* .reset           = */ عدم,
};
