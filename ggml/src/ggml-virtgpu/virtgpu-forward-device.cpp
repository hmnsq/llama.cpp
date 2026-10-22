#ਅਨ੍ਤਰ੍ਭਾਵ "virtgpu-forward-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "virtgpu-shm.h"

ਪੂਰਨ apir_device_get_ਗਿਣਤੀ(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_ਗਿਣਤੀ);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    int32_t dev_ਗਿਣਤੀ = -1;
    apir_decode_int32_t(decoder, &dev_ਗਿਣਤੀ);

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    ਵਾਪਸ dev_ਗਿਣਤੀ;
}

ਅੱਖਰ * apir_device_get_name(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_NAME);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    ਅਚਲ size_t string_ਮਾਪ = apir_decode_array_ਮਾਪ_unchecked(decoder);
    ਅੱਖਰ *       string      = (ਅੱਖਰ *) apir_decoder_alloc_array(ਆਕਾਰ(ਅੱਖਰ), string_ਮਾਪ);
    ਜੇ (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        ਵਾਪਸ ਨਿਰਾ;
    }
    apir_decode_char_array(decoder, string, string_ਮਾਪ);

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    ਵਾਪਸ string;
}

ਅੱਖਰ * apir_device_get_description(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ਅਚਲ size_t string_ਮਾਪ = apir_decode_array_ਮਾਪ_unchecked(decoder);
    ਅੱਖਰ *       string      = (ਅੱਖਰ *) apir_decoder_alloc_array(ਆਕਾਰ(ਅੱਖਰ), string_ਮਾਪ);
    ਜੇ (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device description buffer\n", __func__);

        ਵਾਪਸ ਨਿਰਾ;
    }
    apir_decode_char_array(decoder, string, string_ਮਾਪ);

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    ਵਾਪਸ string;
}

uint32_t apir_device_get_type(virtgpu * gpu) {
    ਸਥਿਰ uint32_t dev_type = 255;
    ਜੇ (dev_type != 255) {
        ਵਾਪਸ dev_type;
    }

    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_uint32_t(decoder, &dev_type);

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    ਵਾਪਸ dev_type;
}

ਖਾਲੀ apir_device_get_ਸਮ੍ਰਿਤੀ(virtgpu * gpu, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
    ਸਥਿਰ size_t         dev_free  = 0;
    ਸਥਿਰ size_t         dev_ਕੁੱਲ = 0;
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_ਸਮ੍ਰਿਤੀ);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_ਮਾਪ_t(decoder, &dev_free);
    apir_decode_ਮਾਪ_t(decoder, &dev_ਕੁੱਲ);

    *ਮੁਕਤ  = dev_free;
    *ਕੁੱਲ = dev_ਕੁੱਲ;

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    ਵਾਪਸ;
}

ਤਾਰਕਿਕ apir_device_supports_op(virtgpu * gpu, ਅਚਲ ggml_tensor * op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ਤਾਰਕਿਕ supports_op;
    apir_decode_bool_t(decoder, &supports_op);

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    ਵਾਪਸ supports_op;
}

apir_ਬਫਰ_type_host_handle_t apir_device_get_ਬਫਰ_type(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_ਬਫਰ_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_ਬਫਰ_type_host_handle_t buft_handle;
    apir_decode_apir_ਬਫਰ_type_host_handle_t(decoder, &buft_handle);

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    ਵਾਪਸ buft_handle;
}

ਖਾਲੀ apir_device_get_props(virtgpu * gpu,
                           ਤਾਰਕਿਕ *    async,
                           ਤਾਰਕਿਕ *    host_ਬਫਰ,
                           ਤਾਰਕਿਕ *    ਬਫਰ_from_host_ptr,
                           ਤਾਰਕਿਕ *    events,
                           ਤਾਰਕਿਕ *    mmap_support) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_PROPS);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_bool_t(decoder, async);
    apir_decode_bool_t(decoder, host_ਬਫਰ);
    apir_decode_bool_t(decoder, ਬਫਰ_from_host_ptr);
    apir_decode_bool_t(decoder, events);
    apir_decode_bool_t(decoder, mmap_support);

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    ਵਾਪਸ;
}

apir_ਬਫਰ_context_t apir_device_ਬਫਰ_from_ptr(virtgpu * gpu, size_t size, size_t max_tensor_ਮਾਪ) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    apir_ਬਫਰ_context_t ਬਫਰ_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_ਬਫਰ_FROM_PTR);

    ਜੇ (virtgpu_shmem_create(gpu, size, &ਬਫਰ_context.shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate %ldb of guest-host shared buffer", __func__, size);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, ਬਫਰ_context.shmem.res_id);

    apir_encode_ਮਾਪ_t(encoder, &size);
    apir_encode_ਮਾਪ_t(encoder, &max_tensor_ਮਾਪ);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_ਬਫਰ_host_handle_t(decoder, &ਬਫਰ_context.host_handle);
    ਬਫਰ_context.buft_host_handle = apir_decode_apir_ਬਫਰ_type_host_handle(decoder);

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    ਵਾਪਸ ਬਫਰ_context;
}
