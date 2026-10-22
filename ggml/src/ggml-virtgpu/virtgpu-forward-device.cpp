#अन्तर्भाव "virtgpu-forward-impl.h"
#अन्तर्भाव "virtgpu-shm.h"

पूर्णाङ्क apir_device_get_गणनम्(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_गणनम्);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    int32_t dev_गणनम् = -1;
    apir_decode_int32_t(decoder, &dev_गणनम्);

    remote_call_finish(gpu, encoder, decoder);

    प्रत्यागम dev_गणनम्;
}

वर्ण * apir_device_get_name(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_NAME);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    ध्रुव size_t string_आकार = apir_decode_array_आकार_unchecked(decoder);
    वर्ण *       string      = (वर्ण *) apir_decoder_alloc_array(परिमाण(वर्ण), string_आकार);
    यदि (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        प्रत्यागम रिक्त;
    }
    apir_decode_char_array(decoder, string, string_आकार);

    remote_call_finish(gpu, encoder, decoder);

    प्रत्यागम string;
}

वर्ण * apir_device_get_description(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ध्रुव size_t string_आकार = apir_decode_array_आकार_unchecked(decoder);
    वर्ण *       string      = (वर्ण *) apir_decoder_alloc_array(परिमाण(वर्ण), string_आकार);
    यदि (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device description buffer\n", __func__);

        प्रत्यागम रिक्त;
    }
    apir_decode_char_array(decoder, string, string_आकार);

    remote_call_finish(gpu, encoder, decoder);

    प्रत्यागम string;
}

uint32_t apir_device_get_type(virtgpu * gpu) {
    स्थिर uint32_t dev_type = 255;
    यदि (dev_type != 255) {
        प्रत्यागम dev_type;
    }

    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_uint32_t(decoder, &dev_type);

    remote_call_finish(gpu, encoder, decoder);

    प्रत्यागम dev_type;
}

शून्य apir_device_get_स्मृतिः(virtgpu * gpu, size_t * मुक्त, size_t * योग) {
    स्थिर size_t         dev_free  = 0;
    स्थिर size_t         dev_योग = 0;
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_स्मृतिः);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_आकार_t(decoder, &dev_free);
    apir_decode_आकार_t(decoder, &dev_योग);

    *मुक्त  = dev_free;
    *योग = dev_योग;

    remote_call_finish(gpu, encoder, decoder);

    प्रत्यागम;
}

तार्किक apir_device_supports_op(virtgpu * gpu, ध्रुव ggml_tensor * op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    तार्किक supports_op;
    apir_decode_bool_t(decoder, &supports_op);

    remote_call_finish(gpu, encoder, decoder);

    प्रत्यागम supports_op;
}

apir_बफर_type_host_handle_t apir_device_get_बफर_type(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_बफर_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_बफर_type_host_handle_t buft_handle;
    apir_decode_apir_बफर_type_host_handle_t(decoder, &buft_handle);

    remote_call_finish(gpu, encoder, decoder);

    प्रत्यागम buft_handle;
}

शून्य apir_device_get_props(virtgpu * gpu,
                           तार्किक *    async,
                           तार्किक *    host_बफर,
                           तार्किक *    बफर_from_host_ptr,
                           तार्किक *    events,
                           तार्किक *    mmap_support) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_PROPS);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_bool_t(decoder, async);
    apir_decode_bool_t(decoder, host_बफर);
    apir_decode_bool_t(decoder, बफर_from_host_ptr);
    apir_decode_bool_t(decoder, events);
    apir_decode_bool_t(decoder, mmap_support);

    remote_call_finish(gpu, encoder, decoder);

    प्रत्यागम;
}

apir_बफर_context_t apir_device_बफर_from_ptr(virtgpu * gpu, size_t size, size_t max_tensor_आकार) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    apir_बफर_context_t बफर_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_बफर_FROM_PTR);

    यदि (virtgpu_shmem_create(gpu, size, &बफर_context.shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate %ldb of guest-host shared buffer", __func__, size);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, बफर_context.shmem.res_id);

    apir_encode_आकार_t(encoder, &size);
    apir_encode_आकार_t(encoder, &max_tensor_आकार);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_बफर_host_handle_t(decoder, &बफर_context.host_handle);
    बफर_context.buft_host_handle = apir_decode_apir_बफर_type_host_handle(decoder);

    remote_call_finish(gpu, encoder, decoder);

    प्रत्यागम बफर_context;
}
