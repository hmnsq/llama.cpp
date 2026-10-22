#समावेश "virtgpu-forward-impl.h"
#समावेश "virtgpu-shm.h"

पूर्णांक apir_device_get_गिनती(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_गिनती);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    int32_t dev_गिनती = -1;
    apir_decode_int32_t(decoder, &dev_गिनती);

    remote_call_finish(gpu, encoder, decoder);

    वापस dev_गिनती;
}

अक्षर * apir_device_get_name(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_NAME);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    स्थिर माप_प्रकार string_आकार = apir_decode_array_आकार_unchecked(decoder);
    अक्षर *       string      = (अक्षर *) apir_decoder_alloc_array(माप(अक्षर), string_आकार);
    अगर (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        वापस शून्य;
    }
    apir_decode_char_array(decoder, string, string_आकार);

    remote_call_finish(gpu, encoder, decoder);

    वापस string;
}

अक्षर * apir_device_get_description(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    स्थिर माप_प्रकार string_आकार = apir_decode_array_आकार_unchecked(decoder);
    अक्षर *       string      = (अक्षर *) apir_decoder_alloc_array(माप(अक्षर), string_आकार);
    अगर (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device description buffer\n", __func__);

        वापस शून्य;
    }
    apir_decode_char_array(decoder, string, string_आकार);

    remote_call_finish(gpu, encoder, decoder);

    वापस string;
}

uint32_t apir_device_get_type(virtgpu * gpu) {
    अटल uint32_t dev_type = 255;
    अगर (dev_type != 255) {
        वापस dev_type;
    }

    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_uint32_t(decoder, &dev_type);

    remote_call_finish(gpu, encoder, decoder);

    वापस dev_type;
}

व्योम apir_device_get_memory(virtgpu * gpu, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
    अटल माप_प्रकार         dev_free  = 0;
    अटल माप_प्रकार         dev_योग = 0;
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_MEMORY);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_आकार_t(decoder, &dev_free);
    apir_decode_आकार_t(decoder, &dev_योग);

    *मुक्त  = dev_free;
    *योग = dev_योग;

    remote_call_finish(gpu, encoder, decoder);

    वापस;
}

bool apir_device_supports_op(virtgpu * gpu, स्थिर ggml_tensor * op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    bool supports_op;
    apir_decode_bool_t(decoder, &supports_op);

    remote_call_finish(gpu, encoder, decoder);

    वापस supports_op;
}

apir_बफर_type_host_handle_t apir_device_get_बफर_type(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_बफर_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_बफर_type_host_handle_t buft_handle;
    apir_decode_apir_बफर_type_host_handle_t(decoder, &buft_handle);

    remote_call_finish(gpu, encoder, decoder);

    वापस buft_handle;
}

व्योम apir_device_get_props(virtgpu * gpu,
                           bool *    async,
                           bool *    host_बफर,
                           bool *    बफर_from_host_ptr,
                           bool *    events,
                           bool *    mmap_support) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_PROPS);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_bool_t(decoder, async);
    apir_decode_bool_t(decoder, host_बफर);
    apir_decode_bool_t(decoder, बफर_from_host_ptr);
    apir_decode_bool_t(decoder, events);
    apir_decode_bool_t(decoder, mmap_support);

    remote_call_finish(gpu, encoder, decoder);

    वापस;
}

apir_बफर_context_t apir_device_बफर_from_ptr(virtgpu * gpu, माप_प्रकार size, माप_प्रकार max_tensor_आकार) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    apir_बफर_context_t बफर_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_बफर_FROM_PTR);

    अगर (virtgpu_shmem_create(gpu, size, &बफर_context.shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate %ldb of guest-host shared buffer", __func__, size);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, बफर_context.shmem.res_id);

    apir_encode_आकार_t(encoder, &size);
    apir_encode_आकार_t(encoder, &max_tensor_आकार);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_बफर_host_handle_t(decoder, &बफर_context.host_handle);
    बफर_context.buft_host_handle = apir_decode_apir_बफर_type_host_handle(decoder);

    remote_call_finish(gpu, encoder, decoder);

    वापस बफर_context;
}
