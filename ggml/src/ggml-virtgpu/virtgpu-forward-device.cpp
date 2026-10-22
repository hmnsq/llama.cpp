#अन्तर्भाव "virtgpu-forward-impl.h"
#अन्तर्भाव "virtgpu-shm.h"

पूर्णाङ्क apir_device_get_गन्ती(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_गन्ती);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    int32_t dev_गन्ती = -1;
    apir_decode_int32_t(decoder, &dev_गन्ती);

    remote_call_finish(gpu, encoder, decoder);

    फर्काऊ dev_गन्ती;
}

वर्ण * apir_device_get_name(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_NAME);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    अचर माप_प्रकार string_size = apir_decode_array_size_unchecked(decoder);
    वर्ण *       string      = (वर्ण *) apir_decoder_alloc_array(परिमाण(वर्ण), string_size);
    यदि (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        फर्काऊ रिक्त;
    }
    apir_decode_char_array(decoder, string, string_size);

    remote_call_finish(gpu, encoder, decoder);

    फर्काऊ string;
}

वर्ण * apir_device_get_description(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    अचर माप_प्रकार string_size = apir_decode_array_size_unchecked(decoder);
    वर्ण *       string      = (वर्ण *) apir_decoder_alloc_array(परिमाण(वर्ण), string_size);
    यदि (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device description buffer\n", __func__);

        फर्काऊ रिक्त;
    }
    apir_decode_char_array(decoder, string, string_size);

    remote_call_finish(gpu, encoder, decoder);

    फर्काऊ string;
}

uint32_t apir_device_get_type(virtgpu * gpu) {
    स्थिर uint32_t dev_type = 255;
    यदि (dev_type != 255) {
        फर्काऊ dev_type;
    }

    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_uint32_t(decoder, &dev_type);

    remote_call_finish(gpu, encoder, decoder);

    फर्काऊ dev_type;
}

शून्य apir_device_get_memory(virtgpu * gpu, माप_प्रकार * मुक्त, माप_प्रकार * total) {
    स्थिर माप_प्रकार         dev_free  = 0;
    स्थिर माप_प्रकार         dev_total = 0;
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_MEMORY);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_size_t(decoder, &dev_free);
    apir_decode_size_t(decoder, &dev_total);

    *मुक्त  = dev_free;
    *total = dev_total;

    remote_call_finish(gpu, encoder, decoder);

    फर्काऊ;
}

तार्किक apir_device_supports_op(virtgpu * gpu, अचर ggml_tensor * op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    तार्किक supports_op;
    apir_decode_bool_t(decoder, &supports_op);

    remote_call_finish(gpu, encoder, decoder);

    फर्काऊ supports_op;
}

apir_buffer_type_host_handle_t apir_device_get_buffer_type(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_BUFFER_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_buffer_type_host_handle_t buft_handle;
    apir_decode_apir_buffer_type_host_handle_t(decoder, &buft_handle);

    remote_call_finish(gpu, encoder, decoder);

    फर्काऊ buft_handle;
}

शून्य apir_device_get_props(virtgpu * gpu,
                           तार्किक *    async,
                           तार्किक *    host_buffer,
                           तार्किक *    buffer_from_host_ptr,
                           तार्किक *    events,
                           तार्किक *    mmap_support) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_PROPS);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_bool_t(decoder, async);
    apir_decode_bool_t(decoder, host_buffer);
    apir_decode_bool_t(decoder, buffer_from_host_ptr);
    apir_decode_bool_t(decoder, events);
    apir_decode_bool_t(decoder, mmap_support);

    remote_call_finish(gpu, encoder, decoder);

    फर्काऊ;
}

apir_buffer_context_t apir_device_buffer_from_ptr(virtgpu * gpu, माप_प्रकार size, माप_प्रकार max_tensor_size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    apir_buffer_context_t buffer_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR);

    यदि (virtgpu_shmem_create(gpu, size, &buffer_context.shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate %ldb of guest-host shared buffer", __func__, size);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, buffer_context.shmem.res_id);

    apir_encode_size_t(encoder, &size);
    apir_encode_size_t(encoder, &max_tensor_size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_buffer_host_handle_t(decoder, &buffer_context.host_handle);
    buffer_context.buft_host_handle = apir_decode_apir_buffer_type_host_handle(decoder);

    remote_call_finish(gpu, encoder, decoder);

    फर्काऊ buffer_context;
}
