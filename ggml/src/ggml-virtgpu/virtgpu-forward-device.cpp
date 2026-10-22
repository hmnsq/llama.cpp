#شامل "virtgpu-forward-impl.h"
#شامل "virtgpu-shm.h"

صحیح apir_device_get_گنتی(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_گنتی);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    int32_t dev_گنتی = -1;
    apir_decode_int32_t(decoder, &dev_گنتی);

    remote_call_اختتام(gpu, encoder, decoder);

    واپس dev_گنتی;
}

حرف * apir_device_get_name(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_NAME);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    ثابت size_t string_سائز = apir_decode_array_سائز_unchecked(decoder);
    حرف *       string      = (حرف *) apir_decoder_alloc_array(حجم(حرف), string_سائز);
    اگر (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        واپس تہی;
    }
    apir_decode_char_array(decoder, string, string_سائز);

    remote_call_اختتام(gpu, encoder, decoder);

    واپس string;
}

حرف * apir_device_get_description(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ثابت size_t string_سائز = apir_decode_array_سائز_unchecked(decoder);
    حرف *       string      = (حرف *) apir_decoder_alloc_array(حجم(حرف), string_سائز);
    اگر (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device description buffer\n", __func__);

        واپس تہی;
    }
    apir_decode_char_array(decoder, string, string_سائز);

    remote_call_اختتام(gpu, encoder, decoder);

    واپس string;
}

uint32_t apir_device_get_type(virtgpu * gpu) {
    ساکن uint32_t dev_type = 255;
    اگر (dev_type != 255) {
        واپس dev_type;
    }

    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_uint32_t(decoder, &dev_type);

    remote_call_اختتام(gpu, encoder, decoder);

    واپس dev_type;
}

خالی apir_device_get_حافظہ(virtgpu * gpu, size_t * آزاد, size_t * کل) {
    ساکن size_t         dev_free  = 0;
    ساکن size_t         dev_کل = 0;
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_حافظہ);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_سائز_t(decoder, &dev_free);
    apir_decode_سائز_t(decoder, &dev_کل);

    *آزاد  = dev_free;
    *کل = dev_کل;

    remote_call_اختتام(gpu, encoder, decoder);

    واپس;
}

منطقی apir_device_supports_op(virtgpu * gpu, ثابت ggml_tensor * op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    منطقی supports_op;
    apir_decode_bool_t(decoder, &supports_op);

    remote_call_اختتام(gpu, encoder, decoder);

    واپس supports_op;
}

apir_بفر_type_host_handle_t apir_device_get_بفر_type(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_بفر_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_بفر_type_host_handle_t buft_handle;
    apir_decode_apir_بفر_type_host_handle_t(decoder, &buft_handle);

    remote_call_اختتام(gpu, encoder, decoder);

    واپس buft_handle;
}

خالی apir_device_get_props(virtgpu * gpu,
                           منطقی *    async,
                           منطقی *    host_بفر,
                           منطقی *    بفر_from_host_ptr,
                           منطقی *    events,
                           منطقی *    mmap_support) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_PROPS);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_bool_t(decoder, async);
    apir_decode_bool_t(decoder, host_بفر);
    apir_decode_bool_t(decoder, بفر_from_host_ptr);
    apir_decode_bool_t(decoder, events);
    apir_decode_bool_t(decoder, mmap_support);

    remote_call_اختتام(gpu, encoder, decoder);

    واپس;
}

apir_بفر_context_t apir_device_بفر_from_ptr(virtgpu * gpu, size_t size, size_t max_tensor_سائز) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    apir_بفر_context_t بفر_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_بفر_FROM_PTR);

    اگر (virtgpu_shmem_create(gpu, size, &بفر_context.shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate %ldb of guest-host shared buffer", __func__, size);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, بفر_context.shmem.res_id);

    apir_encode_سائز_t(encoder, &size);
    apir_encode_سائز_t(encoder, &max_tensor_سائز);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_بفر_host_handle_t(decoder, &بفر_context.host_handle);
    بفر_context.buft_host_handle = apir_decode_apir_بفر_type_host_handle(decoder);

    remote_call_اختتام(gpu, encoder, decoder);

    واپس بفر_context;
}
