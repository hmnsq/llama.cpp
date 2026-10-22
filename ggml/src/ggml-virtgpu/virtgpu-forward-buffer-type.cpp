#ضمن "virtgpu-forward-impl.h"

حرف * apir_بفر_type_get_name(virtgpu * gpu, apir_بفر_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ڪوڊ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_TYPE_GET_NAME);

    apir_encode_apir_بفر_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ثابت size_t string_سائيز = apir_decode_array_سائيز_unchecked(decoder);
    حرف *       string      = (حرف *) apir_decoder_alloc_array(حجم(حرف), string_سائيز);
    إذا (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        apir_decoder_سيٽ_fatal(decoder);
    }
    apir_decode_char_array(decoder, string, string_سائيز);

    remote_call_پڄاڻي(gpu, encoder, decoder);

    أرجع string;
}

size_t apir_بفر_type_get_alignment(virtgpu * gpu, apir_بفر_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ڪوڊ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_TYPE_GET_ALIGNMENT);

    apir_encode_apir_بفر_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t alignment;
    apir_decode_سائيز_t(decoder, &alignment);

    remote_call_پڄاڻي(gpu, encoder, decoder);

    أرجع alignment;
}

size_t apir_بفر_type_get_max_سائيز(virtgpu * gpu, apir_بفر_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ڪوڊ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_TYPE_GET_MAX_سائيز);

    apir_encode_apir_بفر_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t max_size;
    apir_decode_سائيز_t(decoder, &max_size);

    remote_call_پڄاڻي(gpu, encoder, decoder);

    أرجع max_size;
}

apir_بفر_context_t apir_بفر_type_alloc_بفر(virtgpu *                      gpu,
                                                    apir_بفر_type_host_handle_t host_handle,
                                                    size_t                         size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ڪوڊ ret;

    apir_بفر_context_t بفر_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_TYPE_ALLOC_بفر);

    apir_encode_apir_بفر_type_host_handle(encoder, host_handle);

    apir_encode_سائيز_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_بفر_host_handle_t(decoder, &بفر_context.host_handle);

    remote_call_پڄاڻي(gpu, encoder, decoder);

    أرجع بفر_context;
}

size_t apir_بفر_type_get_alloc_سائيز(virtgpu *                      gpu,
                                       apir_بفر_type_host_handle_t host_handle,
                                       ثابت ggml_tensor *            op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ڪوڊ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_TYPE_GET_ALLOC_سائيز);

    apir_encode_apir_بفر_type_host_handle(encoder, host_handle);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t alloc_سائيز;
    apir_decode_سائيز_t(decoder, &alloc_سائيز);

    remote_call_پڄاڻي(gpu, encoder, decoder);

    أرجع alloc_سائيز;
}
