#অন্তর্ভাব "virtgpu-forward-impl.h"

অক্ষর * apir_বাফার_type_get_name(virtgpu * gpu, apir_বাফার_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_বাফার_TYPE_GET_NAME);

    apir_encode_apir_বাফার_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ধ্রুবক size_t string_আকার = apir_decode_array_আকার_unchecked(decoder);
    অক্ষর *       string      = (অক্ষর *) apir_decoder_alloc_array(আকার(অক্ষর), string_আকার);
    যদি (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        apir_decoder_set_fatal(decoder);
    }
    apir_decode_char_array(decoder, string, string_আকার);

    remote_call_finish(gpu, encoder, decoder);

    ফেরত string;
}

size_t apir_বাফার_type_get_alignment(virtgpu * gpu, apir_বাফার_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_বাফার_TYPE_GET_ALIGNMENT);

    apir_encode_apir_বাফার_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t alignment;
    apir_decode_আকার_t(decoder, &alignment);

    remote_call_finish(gpu, encoder, decoder);

    ফেরত alignment;
}

size_t apir_বাফার_type_get_max_আকার(virtgpu * gpu, apir_বাফার_type_host_handle_t host_handle) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_বাফার_TYPE_GET_MAX_আকার);

    apir_encode_apir_বাফার_type_host_handle(encoder, host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t max_size;
    apir_decode_আকার_t(decoder, &max_size);

    remote_call_finish(gpu, encoder, decoder);

    ফেরত max_size;
}

apir_বাফার_context_t apir_বাফার_type_alloc_বাফার(virtgpu *                      gpu,
                                                    apir_বাফার_type_host_handle_t host_handle,
                                                    size_t                         size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    apir_বাফার_context_t বাফার_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_বাফার_TYPE_ALLOC_বাফার);

    apir_encode_apir_বাফার_type_host_handle(encoder, host_handle);

    apir_encode_আকার_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_বাফার_host_handle_t(decoder, &বাফার_context.host_handle);

    remote_call_finish(gpu, encoder, decoder);

    ফেরত বাফার_context;
}

size_t apir_বাফার_type_get_alloc_আকার(virtgpu *                      gpu,
                                       apir_বাফার_type_host_handle_t host_handle,
                                       ধ্রুবক ggml_tensor *            op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_বাফার_TYPE_GET_ALLOC_আকার);

    apir_encode_apir_বাফার_type_host_handle(encoder, host_handle);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    size_t alloc_আকার;
    apir_decode_আকার_t(decoder, &alloc_আকার);

    remote_call_finish(gpu, encoder, decoder);

    ফেরত alloc_আকার;
}
