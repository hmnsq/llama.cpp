#ಅನ್ತರ್ಭಾವ "virtgpu-forward-impl.h"

ಅಕ್ಷರ * apir_ಬಫರ್_type_get_name(virtgpu * gpu, apir_ಬಫರ್_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ಬಫರ್_TYPE_GET_NAME);

    apir_encode_apir_ಬಫರ್_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ಸ್ಥಿರಾಂಕ size_t string_ಗಾತ್ರ = apir_decode_array_ಗಾತ್ರ_unchecked(decoder);
    ಅಕ್ಷರ *       string      = (ಅಕ್ಷರ *) apir_decoder_alloc_array(ಗಾತ್ರ(ಅಕ್ಷರ), string_ಗಾತ್ರ);
    ಆದರೆ (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        apir_decoder_ಗಣ_fatal(decoder);
    }
    apir_decode_char_array(decoder, string, string_ಗಾತ್ರ);

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    ಹಿಂತಿರುಗಿ string;
}

size_t apir_ಬಫರ್_type_get_alignment(virtgpu * gpu, apir_ಬಫರ್_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ಬಫರ್_TYPE_GET_ALIGNMENT);

    apir_encode_apir_ಬಫರ್_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t alignment;
    apir_decode_ಗಾತ್ರ_t(decoder, &alignment);

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    ಹಿಂತಿರುಗಿ alignment;
}

size_t apir_ಬಫರ್_type_get_max_ಗಾತ್ರ(virtgpu * gpu, apir_ಬಫರ್_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ಬಫರ್_TYPE_GET_MAX_ಗಾತ್ರ);

    apir_encode_apir_ಬಫರ್_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t max_size;
    apir_decode_ಗಾತ್ರ_t(decoder, &max_size);

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    ಹಿಂತಿರುಗಿ max_size;
}

apir_ಬಫರ್_context_t apir_ಬಫರ್_type_alloc_ಬಫರ್(virtgpu *                      gpu,
                                                    apir_ಬಫರ್_type_host_handle_t host_handle,
                                                    size_t                         size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    apir_ಬಫರ್_context_t ಬಫರ್_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ಬಫರ್_TYPE_ALLOC_ಬಫರ್);

    apir_encode_apir_ಬಫರ್_type_host_handle(encoder, host_handle);

    apir_encode_ಗಾತ್ರ_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_ಬಫರ್_host_handle_t(decoder, &ಬಫರ್_context.host_handle);

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    ಹಿಂತಿರುಗಿ ಬಫರ್_context;
}

size_t apir_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(virtgpu *                      gpu,
                                       apir_ಬಫರ್_type_host_handle_t host_handle,
                                       ಸ್ಥಿರಾಂಕ ggml_tensor *            op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ಬಫರ್_TYPE_GET_ALLOC_ಗಾತ್ರ);

    apir_encode_apir_ಬಫರ್_type_host_handle(encoder, host_handle);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t alloc_ಗಾತ್ರ;
    apir_decode_ಗಾತ್ರ_t(decoder, &alloc_ಗಾತ್ರ);

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    ಹಿಂತಿರುಗಿ alloc_ಗಾತ್ರ;
}
