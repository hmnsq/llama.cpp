#অন্তর্ভাব "ggml-remoting.h"

#define বাফার_TO_GPU(name) ((ggml_backend_remoting_বাফার_context *) (name)->context)->gpu

স্থির শূন্য * ggml_backend_remoting_বাফার_get_base(ggml_backend_বাফার_t বাফার) {
    ggml_backend_remoting_বাফার_context * context = (ggml_backend_remoting_বাফার_context *) বাফার->context;
    যদি (context->base) {
        ফেরত context->base;
    }

    context->base = apir_বাফার_get_base(বাফার_TO_GPU(বাফার), বাফার_TO_APIR_CONTEXT(বাফার));

    ফেরত context->base;
}

স্থির শূন্য ggml_backend_remoting_বাফার_set_tensor(ggml_backend_বাফার_t বাফার,
                                                    ggml_tensor *         tensor,
                                                    ধ্রুবক শূন্য *          তথ্য,
                                                    size_t                offset,
                                                    size_t                size) {
    virtgpu * gpu = বাফার_TO_GPU(বাফার);

    ggml_backend_remoting_বাফার_context * context = বাফার_TO_GGML_CONTEXT(বাফার);
    যদি (context->is_from_ptr) {
        memcpy((অক্ষর *) tensor->তথ্য + offset, তথ্য, size);
    } অন্যথা {
        apir_বাফার_set_tensor(gpu, বাফার_TO_APIR_CONTEXT(বাফার), tensor, তথ্য, offset, size);
    }

    ফেরত;
}

স্থির শূন্য ggml_backend_remoting_বাফার_get_tensor(ggml_backend_বাফার_t বাফার,
                                                    ধ্রুবক ggml_tensor *   tensor,
                                                    শূন্য *                তথ্য,
                                                    size_t                offset,
                                                    size_t                size) {
    virtgpu *                              gpu     = বাফার_TO_GPU(বাফার);
    ggml_backend_remoting_বাফার_context * context = বাফার_TO_GGML_CONTEXT(বাফার);
    যদি (context->is_from_ptr) {
        memcpy(তথ্য, (ধ্রুবক অক্ষর *) tensor->তথ্য + offset, size);
    } অন্যথা {
        apir_বাফার_get_tensor(gpu, বাফার_TO_APIR_CONTEXT(বাফার), tensor, তথ্য, offset, size);
    }
}

স্থির শূন্য ggml_backend_remoting_বাফার_set_tensor_from_ptr(ggml_backend_বাফার_t বাফার,
                                                             ggml_tensor *         tensor,
                                                             ধ্রুবক শূন্য *          তথ্য,
                                                             size_t                offset,
                                                             size_t                size) {
    UNUSED(বাফার);

    memcpy((অক্ষর *) tensor->তথ্য + offset, তথ্য, size);

    ফেরত;
}

স্থির শূন্য ggml_backend_remoting_বাফার_get_tensor_from_ptr(ggml_backend_বাফার_t বাফার,
                                                             ধ্রুবক ggml_tensor *   tensor,
                                                             শূন্য *                তথ্য,
                                                             size_t                offset,
                                                             size_t                size) {
    UNUSED(বাফার);

    memcpy(তথ্য, (ধ্রুবক অক্ষর *) tensor->তথ্য + offset, size);
}

স্থির যৌক্তিক ggml_backend_remoting_বাফার_cpy_tensor(ggml_backend_বাফার_t বাফার,
                                                    ধ্রুবক ggml_tensor *   src,
                                                    ggml_tensor *         dst) {
    virtgpu * gpu = বাফার_TO_GPU(বাফার);

    যৌক্তিক ret = apir_বাফার_cpy_tensor(gpu, বাফার_TO_APIR_CONTEXT(বাফার), src, dst);

    ফেরত ret;
}

স্থির শূন্য ggml_backend_remoting_বাফার_clear(ggml_backend_বাফার_t বাফার, uint8_t value) {
    virtgpu * gpu = বাফার_TO_GPU(বাফার);

    apir_বাফার_clear(gpu, বাফার_TO_APIR_CONTEXT(বাফার), value);

    ফেরত;
}

স্থির শূন্য ggml_backend_remoting_বাফার_free_বাফার(ggml_backend_বাফার_t বাফার) {
    virtgpu * gpu = বাফার_TO_GPU(বাফার);

    apir_বাফার_free_বাফার(gpu, বাফার_TO_APIR_CONTEXT(বাফার));

    ggml_backend_remoting_বাফার_context * context = বাফার_TO_GGML_CONTEXT(বাফার);
    মুক্ত(context);
    বাফার->context = ফাঁকা;
}

ধ্রুবক ggml_backend_বাফার_i ggml_backend_remoting_বাফার_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_বাফার_free_বাফার,
    /* .get_base        = */ ggml_backend_remoting_বাফার_get_base,
    /* .init_tensor     = */ ফাঁকা,
    /* .memset_tensor   = */ ফাঁকা,
    /* .set_tensor      = */ ggml_backend_remoting_বাফার_set_tensor,
    /* .get_tensor      = */ ggml_backend_remoting_বাফার_get_tensor,
    /* .set_tensor_2d   = */ ফাঁকা,
    /* .get_tensor_2d   = */ ফাঁকা,
    /* .cpy_tensor      = */ ggml_backend_remoting_বাফার_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_বাফার_clear,
    /* .reset           = */ ফাঁকা,
};

ধ্রুবক ggml_backend_বাফার_i ggml_backend_remoting_বাফার_from_ptr_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_বাফার_free_বাফার,
    /* .get_base        = */ ggml_backend_remoting_বাফার_get_base,
    /* .init_tensor     = */ ফাঁকা,
    /* .memset_tensor   = */ ফাঁকা,
    /* .set_tensor      = */ ggml_backend_remoting_বাফার_set_tensor_from_ptr,
    /* .get_tensor      = */ ggml_backend_remoting_বাফার_get_tensor_from_ptr,
    /* .set_tensor_2d   = */ ফাঁকা,
    /* .get_tensor_2d   = */ ফাঁকা,
    /* .cpy_tensor      = */ ggml_backend_remoting_বাফার_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_বাফার_clear,
    /* .reset           = */ ফাঁকা,
};
