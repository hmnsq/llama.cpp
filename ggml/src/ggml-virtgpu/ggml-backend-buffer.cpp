#అన్తర్భావ "ggml-remoting.h"

#నిర్వచించు బఫరు_గమ్యం_GPU(name) ((ggml_backend_remoting_బఫరు_context *) (name)->context)->gpu

స్థిర శూన్య * ggml_backend_remoting_బఫరు_get_పునాది(ggml_backend_బఫరు_t బఫరు) {
    ggml_backend_remoting_బఫరు_context * context = (ggml_backend_remoting_బఫరు_context *) బఫరు->context;
    ఒకవేళ (context->పునాది) {
        తిరిగి context->పునాది;
    }

    context->పునాది = apir_బఫరు_get_పునాది(బఫరు_గమ్యం_GPU(బఫరు), బఫరు_గమ్యం_APIR_CONTEXT(బఫరు));

    తిరిగి context->పునాది;
}

స్థిర శూన్య ggml_backend_remoting_బఫరు_సమితి_tensor(ggml_backend_బఫరు_t బఫరు,
                                                    ggml_tensor *         tensor,
                                                    స్థిరరాశి శూన్య *          దత్తాంశం,
                                                    size_t                స్థానభ్రంశం,
                                                    size_t                size) {
    virtgpu * gpu = బఫరు_గమ్యం_GPU(బఫరు);

    ggml_backend_remoting_బఫరు_context * context = బఫరు_గమ్యం_GGML_CONTEXT(బఫరు);
    ఒకవేళ (context->is_from_ptr) {
        జ్ఞాపకం_నకలు((అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, దత్తాంశం, size);
    } లేకపోతే {
        apir_బఫరు_సమితి_tensor(gpu, బఫరు_గమ్యం_APIR_CONTEXT(బఫరు), tensor, దత్తాంశం, స్థానభ్రంశం, size);
    }

    తిరిగి;
}

స్థిర శూన్య ggml_backend_remoting_బఫరు_get_tensor(ggml_backend_బఫరు_t బఫరు,
                                                    స్థిరరాశి ggml_tensor *   tensor,
                                                    శూన్య *                దత్తాంశం,
                                                    size_t                స్థానభ్రంశం,
                                                    size_t                size) {
    virtgpu *                              gpu     = బఫరు_గమ్యం_GPU(బఫరు);
    ggml_backend_remoting_బఫరు_context * context = బఫరు_గమ్యం_GGML_CONTEXT(బఫరు);
    ఒకవేళ (context->is_from_ptr) {
        జ్ఞాపకం_నకలు(దత్తాంశం, (స్థిరరాశి అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, size);
    } లేకపోతే {
        apir_బఫరు_get_tensor(gpu, బఫరు_గమ్యం_APIR_CONTEXT(బఫరు), tensor, దత్తాంశం, స్థానభ్రంశం, size);
    }
}

స్థిర శూన్య ggml_backend_remoting_బఫరు_సమితి_tensor_from_ptr(ggml_backend_బఫరు_t బఫరు,
                                                             ggml_tensor *         tensor,
                                                             స్థిరరాశి శూన్య *          దత్తాంశం,
                                                             size_t                స్థానభ్రంశం,
                                                             size_t                size) {
    వాడని(బఫరు);

    జ్ఞాపకం_నకలు((అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, దత్తాంశం, size);

    తిరిగి;
}

స్థిర శూన్య ggml_backend_remoting_బఫరు_get_tensor_from_ptr(ggml_backend_బఫరు_t బఫరు,
                                                             స్థిరరాశి ggml_tensor *   tensor,
                                                             శూన్య *                దత్తాంశం,
                                                             size_t                స్థానభ్రంశం,
                                                             size_t                size) {
    వాడని(బఫరు);

    జ్ఞాపకం_నకలు(దత్తాంశం, (స్థిరరాశి అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, size);
}

స్థిర తార్కిక ggml_backend_remoting_బఫరు_cpy_tensor(ggml_backend_బఫరు_t బఫరు,
                                                    స్థిరరాశి ggml_tensor *   src,
                                                    ggml_tensor *         dst) {
    virtgpu * gpu = బఫరు_గమ్యం_GPU(బఫరు);

    తార్కిక ret = apir_బఫరు_cpy_tensor(gpu, బఫరు_గమ్యం_APIR_CONTEXT(బఫరు), src, dst);

    తిరిగి ret;
}

స్థిర శూన్య ggml_backend_remoting_బఫరు_clear(ggml_backend_బఫరు_t బఫరు, uint8_t value) {
    virtgpu * gpu = బఫరు_గమ్యం_GPU(బఫరు);

    apir_బఫరు_clear(gpu, బఫరు_గమ్యం_APIR_CONTEXT(బఫరు), value);

    తిరిగి;
}

స్థిర శూన్య ggml_backend_remoting_బఫరు_free_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    virtgpu * gpu = బఫరు_గమ్యం_GPU(బఫరు);

    apir_బఫరు_free_బఫరు(gpu, బఫరు_గమ్యం_APIR_CONTEXT(బఫరు));

    ggml_backend_remoting_బఫరు_context * context = బఫరు_గమ్యం_GGML_CONTEXT(బఫరు);
    విడుదల(context);
    బఫరు->context = ఖాళీ;
}

స్థిరరాశి ggml_backend_బఫరు_i ggml_backend_remoting_బఫరు_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_బఫరు_free_బఫరు,
    /* .get_base        = */ ggml_backend_remoting_బఫరు_get_పునాది,
    /* .init_tensor     = */ ఖాళీ,
    /* .memset_tensor   = */ ఖాళీ,
    /* .set_tensor      = */ ggml_backend_remoting_బఫరు_సమితి_tensor,
    /* .get_tensor      = */ ggml_backend_remoting_బఫరు_get_tensor,
    /* .set_tensor_2d   = */ ఖాళీ,
    /* .get_tensor_2d   = */ ఖాళీ,
    /* .cpy_tensor      = */ ggml_backend_remoting_బఫరు_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_బఫరు_clear,
    /* .reset           = */ ఖాళీ,
};

స్థిరరాశి ggml_backend_బఫరు_i ggml_backend_remoting_బఫరు_from_ptr_interface = {
    /* .free_buffer     = */ ggml_backend_remoting_బఫరు_free_బఫరు,
    /* .get_base        = */ ggml_backend_remoting_బఫరు_get_పునాది,
    /* .init_tensor     = */ ఖాళీ,
    /* .memset_tensor   = */ ఖాళీ,
    /* .set_tensor      = */ ggml_backend_remoting_బఫరు_సమితి_tensor_from_ptr,
    /* .get_tensor      = */ ggml_backend_remoting_బఫరు_get_tensor_from_ptr,
    /* .set_tensor_2d   = */ ఖాళీ,
    /* .get_tensor_2d   = */ ఖాళీ,
    /* .cpy_tensor      = */ ggml_backend_remoting_బఫరు_cpy_tensor,
    /* .clear           = */ ggml_backend_remoting_బఫరు_clear,
    /* .reset           = */ ఖాళీ,
};
