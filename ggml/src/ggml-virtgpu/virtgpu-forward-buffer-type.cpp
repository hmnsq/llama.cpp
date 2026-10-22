#ضمن "virtgpu-forward-impl.h"

حرف * apir_میانگیر_type_get_name(virtgpu * gpu, apir_میانگیر_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_میانگیر_TYPE_GET_NAME);

    apir_encode_apir_میانگیر_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ثابت size_t string_اندازه = apir_decode_array_اندازه_unchecked(decoder);
    حرف *       string      = (حرف *) apir_decoder_alloc_array(حجم(حرف), string_اندازه);
    إذا (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        apir_decoder_مجموعه_fatal(decoder);
    }
    apir_decode_char_array(decoder, string, string_اندازه);

    remote_call_پایان(gpu, encoder, decoder);

    أرجع string;
}

size_t apir_میانگیر_type_get_alignment(virtgpu * gpu, apir_میانگیر_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_میانگیر_TYPE_GET_ALIGNMENT);

    apir_encode_apir_میانگیر_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t alignment;
    apir_decode_اندازه_t(decoder, &alignment);

    remote_call_پایان(gpu, encoder, decoder);

    أرجع alignment;
}

size_t apir_میانگیر_type_get_max_اندازه(virtgpu * gpu, apir_میانگیر_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_میانگیر_TYPE_GET_MAX_اندازه);

    apir_encode_apir_میانگیر_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t max_size;
    apir_decode_اندازه_t(decoder, &max_size);

    remote_call_پایان(gpu, encoder, decoder);

    أرجع max_size;
}

apir_میانگیر_context_t apir_میانگیر_type_alloc_میانگیر(virtgpu *                      gpu,
                                                    apir_میانگیر_type_host_handle_t host_handle,
                                                    size_t                         size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    apir_میانگیر_context_t میانگیر_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_میانگیر_TYPE_ALLOC_میانگیر);

    apir_encode_apir_میانگیر_type_host_handle(encoder, host_handle);

    apir_encode_اندازه_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_میانگیر_host_handle_t(decoder, &میانگیر_context.host_handle);

    remote_call_پایان(gpu, encoder, decoder);

    أرجع میانگیر_context;
}

size_t apir_میانگیر_type_get_alloc_اندازه(virtgpu *                      gpu,
                                       apir_میانگیر_type_host_handle_t host_handle,
                                       ثابت ggml_tensor *            op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_میانگیر_TYPE_GET_ALLOC_اندازه);

    apir_encode_apir_میانگیر_type_host_handle(encoder, host_handle);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t alloc_اندازه;
    apir_decode_اندازه_t(decoder, &alloc_اندازه);

    remote_call_پایان(gpu, encoder, decoder);

    أرجع alloc_اندازه;
}
