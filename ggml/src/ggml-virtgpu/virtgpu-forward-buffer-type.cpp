#شامل "virtgpu-forward-impl.h"

حرف * apir_بفر_type_get_name(virtgpu * gpu, apir_بفر_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_TYPE_GET_NAME);

    apir_encode_apir_بفر_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ثابت size_t string_سائز = apir_decode_array_سائز_unchecked(decoder);
    حرف *       string      = (حرف *) apir_decoder_alloc_array(حجم(حرف), string_سائز);
    اگر (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        apir_decoder_سیٹ_fatal(decoder);
    }
    apir_decode_char_array(decoder, string, string_سائز);

    remote_call_اختتام(gpu, encoder, decoder);

    واپس string;
}

size_t apir_بفر_type_get_alignment(virtgpu * gpu, apir_بفر_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_TYPE_GET_ALIGNMENT);

    apir_encode_apir_بفر_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t alignment;
    apir_decode_سائز_t(decoder, &alignment);

    remote_call_اختتام(gpu, encoder, decoder);

    واپس alignment;
}

size_t apir_بفر_type_get_max_سائز(virtgpu * gpu, apir_بفر_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_TYPE_GET_MAX_سائز);

    apir_encode_apir_بفر_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t max_size;
    apir_decode_سائز_t(decoder, &max_size);

    remote_call_اختتام(gpu, encoder, decoder);

    واپس max_size;
}

apir_بفر_context_t apir_بفر_type_alloc_بفر(virtgpu *                      gpu,
                                                    apir_بفر_type_host_handle_t host_handle,
                                                    size_t                         size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    apir_بفر_context_t بفر_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_TYPE_ALLOC_بفر);

    apir_encode_apir_بفر_type_host_handle(encoder, host_handle);

    apir_encode_سائز_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_بفر_host_handle_t(decoder, &بفر_context.host_handle);

    remote_call_اختتام(gpu, encoder, decoder);

    واپس بفر_context;
}

size_t apir_بفر_type_get_alloc_سائز(virtgpu *                      gpu,
                                       apir_بفر_type_host_handle_t host_handle,
                                       ثابت ggml_tensor *            op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_TYPE_GET_ALLOC_سائز);

    apir_encode_apir_بفر_type_host_handle(encoder, host_handle);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t alloc_سائز;
    apir_decode_سائز_t(decoder, &alloc_سائز);

    remote_call_اختتام(gpu, encoder, decoder);

    واپس alloc_سائز;
}
