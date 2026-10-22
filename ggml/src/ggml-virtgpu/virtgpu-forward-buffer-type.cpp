#అన్తర్భావ "virtgpu-forward-impl.h"

అక్షరం * apir_బఫరు_type_get_name(virtgpu * gpu, apir_బఫరు_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_బఫరు_TYPE_GET_NAME);

    apir_encode_apir_బఫరు_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    స్థిరరాశి size_t string_కొలత = apir_decode_array_కొలత_unchecked(decoder);
    అక్షరం *       string      = (అక్షరం *) apir_decoder_alloc_array(పరిమాణం(అక్షరం), string_కొలత);
    ఒకవేళ (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        apir_decoder_సమితి_fatal(decoder);
    }
    apir_decode_char_array(decoder, string, string_కొలత);

    remote_call_ముగింపు(gpu, encoder, decoder);

    తిరిగి string;
}

size_t apir_బఫరు_type_get_alignment(virtgpu * gpu, apir_బఫరు_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_బఫరు_TYPE_GET_ALIGNMENT);

    apir_encode_apir_బఫరు_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t alignment;
    apir_decode_కొలత_t(decoder, &alignment);

    remote_call_ముగింపు(gpu, encoder, decoder);

    తిరిగి alignment;
}

size_t apir_బఫరు_type_get_max_కొలత(virtgpu * gpu, apir_బఫరు_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_బఫరు_TYPE_GET_MAX_కొలత);

    apir_encode_apir_బఫరు_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t max_size;
    apir_decode_కొలత_t(decoder, &max_size);

    remote_call_ముగింపు(gpu, encoder, decoder);

    తిరిగి max_size;
}

apir_బఫరు_context_t apir_బఫరు_type_alloc_బఫరు(virtgpu *                      gpu,
                                                    apir_బఫరు_type_host_handle_t host_handle,
                                                    size_t                         size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    apir_బఫరు_context_t బఫరు_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_బఫరు_TYPE_ALLOC_బఫరు);

    apir_encode_apir_బఫరు_type_host_handle(encoder, host_handle);

    apir_encode_కొలత_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_బఫరు_host_handle_t(decoder, &బఫరు_context.host_handle);

    remote_call_ముగింపు(gpu, encoder, decoder);

    తిరిగి బఫరు_context;
}

size_t apir_బఫరు_type_get_alloc_కొలత(virtgpu *                      gpu,
                                       apir_బఫరు_type_host_handle_t host_handle,
                                       స్థిరరాశి ggml_tensor *            op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_బఫరు_TYPE_GET_ALLOC_కొలత);

    apir_encode_apir_బఫరు_type_host_handle(encoder, host_handle);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t alloc_కొలత;
    apir_decode_కొలత_t(decoder, &alloc_కొలత);

    remote_call_ముగింపు(gpu, encoder, decoder);

    తిరిగి alloc_కొలత;
}
