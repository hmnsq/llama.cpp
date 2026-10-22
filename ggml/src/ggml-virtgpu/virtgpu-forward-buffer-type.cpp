#अन्तर्भाव "virtgpu-forward-impl.h"

वर्ण * apir_बफर_type_get_name(virtgpu * gpu, apir_बफर_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_TYPE_GET_NAME);

    apir_encode_apir_बफर_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ध्रुव size_t string_आकार = apir_decode_array_आकार_unchecked(decoder);
    वर्ण *       string      = (वर्ण *) apir_decoder_alloc_array(परिमाण(वर्ण), string_आकार);
    यदि (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        apir_decoder_समुच्चयः_fatal(decoder);
    }
    apir_decode_char_array(decoder, string, string_आकार);

    remote_call_finish(gpu, encoder, decoder);

    प्रत्यागम string;
}

size_t apir_बफर_type_get_alignment(virtgpu * gpu, apir_बफर_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_TYPE_GET_ALIGNMENT);

    apir_encode_apir_बफर_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t alignment;
    apir_decode_आकार_t(decoder, &alignment);

    remote_call_finish(gpu, encoder, decoder);

    प्रत्यागम alignment;
}

size_t apir_बफर_type_get_max_आकार(virtgpu * gpu, apir_बफर_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_TYPE_GET_MAX_आकार);

    apir_encode_apir_बफर_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t max_size;
    apir_decode_आकार_t(decoder, &max_size);

    remote_call_finish(gpu, encoder, decoder);

    प्रत्यागम max_size;
}

apir_बफर_context_t apir_बफर_type_alloc_बफर(virtgpu *                      gpu,
                                                    apir_बफर_type_host_handle_t host_handle,
                                                    size_t                         size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    apir_बफर_context_t बफर_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_TYPE_ALLOC_बफर);

    apir_encode_apir_बफर_type_host_handle(encoder, host_handle);

    apir_encode_आकार_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_बफर_host_handle_t(decoder, &बफर_context.host_handle);

    remote_call_finish(gpu, encoder, decoder);

    प्रत्यागम बफर_context;
}

size_t apir_बफर_type_get_alloc_आकार(virtgpu *                      gpu,
                                       apir_बफर_type_host_handle_t host_handle,
                                       ध्रुव ggml_tensor *            op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_TYPE_GET_ALLOC_आकार);

    apir_encode_apir_बफर_type_host_handle(encoder, host_handle);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t alloc_आकार;
    apir_decode_आकार_t(decoder, &alloc_आकार);

    remote_call_finish(gpu, encoder, decoder);

    प्रत्यागम alloc_आकार;
}
