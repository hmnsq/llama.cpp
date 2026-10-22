#ਅਨ੍ਤਰ੍ਭਾਵ "virtgpu-forward-impl.h"

ਅੱਖਰ * apir_ਬਫਰ_type_get_name(virtgpu * gpu, apir_ਬਫਰ_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ਬਫਰ_TYPE_GET_NAME);

    apir_encode_apir_ਬਫਰ_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ਅਚਲ size_t string_ਮਾਪ = apir_decode_array_ਮਾਪ_unchecked(decoder);
    ਅੱਖਰ *       string      = (ਅੱਖਰ *) apir_decoder_alloc_array(ਆਕਾਰ(ਅੱਖਰ), string_ਮਾਪ);
    ਜੇ (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        apir_decoder_ਸੈੱਟ_fatal(decoder);
    }
    apir_decode_char_array(decoder, string, string_ਮਾਪ);

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    ਵਾਪਸ string;
}

size_t apir_ਬਫਰ_type_get_alignment(virtgpu * gpu, apir_ਬਫਰ_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ਬਫਰ_TYPE_GET_ALIGNMENT);

    apir_encode_apir_ਬਫਰ_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t alignment;
    apir_decode_ਮਾਪ_t(decoder, &alignment);

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    ਵਾਪਸ alignment;
}

size_t apir_ਬਫਰ_type_get_max_ਮਾਪ(virtgpu * gpu, apir_ਬਫਰ_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ਬਫਰ_TYPE_GET_MAX_ਮਾਪ);

    apir_encode_apir_ਬਫਰ_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t max_size;
    apir_decode_ਮਾਪ_t(decoder, &max_size);

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    ਵਾਪਸ max_size;
}

apir_ਬਫਰ_context_t apir_ਬਫਰ_type_alloc_ਬਫਰ(virtgpu *                      gpu,
                                                    apir_ਬਫਰ_type_host_handle_t host_handle,
                                                    size_t                         size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    apir_ਬਫਰ_context_t ਬਫਰ_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ਬਫਰ_TYPE_ALLOC_ਬਫਰ);

    apir_encode_apir_ਬਫਰ_type_host_handle(encoder, host_handle);

    apir_encode_ਮਾਪ_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_ਬਫਰ_host_handle_t(decoder, &ਬਫਰ_context.host_handle);

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    ਵਾਪਸ ਬਫਰ_context;
}

size_t apir_ਬਫਰ_type_get_alloc_ਮਾਪ(virtgpu *                      gpu,
                                       apir_ਬਫਰ_type_host_handle_t host_handle,
                                       ਅਚਲ ggml_tensor *            op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ਬਫਰ_TYPE_GET_ALLOC_ਮਾਪ);

    apir_encode_apir_ਬਫਰ_type_host_handle(encoder, host_handle);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t alloc_ਮਾਪ;
    apir_decode_ਮਾਪ_t(decoder, &alloc_ਮਾਪ);

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    ਵਾਪਸ alloc_ਮਾਪ;
}
