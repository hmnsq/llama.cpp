#समावेश "virtgpu-forward-impl.h"

अक्षर * apir_बफर_type_get_name(virtgpu * gpu, apir_बफर_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_TYPE_GET_NAME);

    apir_encode_apir_बफर_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    स्थिर माप_प्रकार string_आकार = apir_decode_array_आकार_unchecked(decoder);
    अक्षर *       string      = (अक्षर *) apir_decoder_alloc_array(माप(अक्षर), string_आकार);
    अगर (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        apir_decoder_set_fatal(decoder);
    }
    apir_decode_char_array(decoder, string, string_आकार);

    remote_call_finish(gpu, encoder, decoder);

    वापस string;
}

माप_प्रकार apir_बफर_type_get_alignment(virtgpu * gpu, apir_बफर_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_TYPE_GET_ALIGNMENT);

    apir_encode_apir_बफर_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    माप_प्रकार alignment;
    apir_decode_आकार_t(decoder, &alignment);

    remote_call_finish(gpu, encoder, decoder);

    वापस alignment;
}

माप_प्रकार apir_बफर_type_get_max_आकार(virtgpu * gpu, apir_बफर_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_TYPE_GET_MAX_आकार);

    apir_encode_apir_बफर_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    माप_प्रकार max_size;
    apir_decode_आकार_t(decoder, &max_size);

    remote_call_finish(gpu, encoder, decoder);

    वापस max_size;
}

apir_बफर_context_t apir_बफर_type_alloc_बफर(virtgpu *                      gpu,
                                                    apir_बफर_type_host_handle_t host_handle,
                                                    माप_प्रकार                         size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    apir_बफर_context_t बफर_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_TYPE_ALLOC_बफर);

    apir_encode_apir_बफर_type_host_handle(encoder, host_handle);

    apir_encode_आकार_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_बफर_host_handle_t(decoder, &बफर_context.host_handle);

    remote_call_finish(gpu, encoder, decoder);

    वापस बफर_context;
}

माप_प्रकार apir_बफर_type_get_alloc_आकार(virtgpu *                      gpu,
                                       apir_बफर_type_host_handle_t host_handle,
                                       स्थिर ggml_tensor *            op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_TYPE_GET_ALLOC_आकार);

    apir_encode_apir_बफर_type_host_handle(encoder, host_handle);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    माप_प्रकार alloc_आकार;
    apir_decode_आकार_t(decoder, &alloc_आकार);

    remote_call_finish(gpu, encoder, decoder);

    वापस alloc_आकार;
}
