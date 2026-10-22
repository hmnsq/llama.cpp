#ಅನ್ತರ್ಭಾವ "virtgpu-forward-impl.h"
#ಅನ್ತರ್ಭಾವ "virtgpu-shm.h"

ಪೂರ್ಣಾಙ್ಕ apir_device_get_ಎಣಿಕೆ(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_ಎಣಿಕೆ);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    int32_t dev_ಎಣಿಕೆ = -1;
    apir_decode_int32_t(decoder, &dev_ಎಣಿಕೆ);

    remote_call_finish(gpu, encoder, decoder);

    ಪ್ರತ್ಯಾಗಮ dev_ಎಣಿಕೆ;
}

ವರ್ಣ * apir_device_get_name(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_NAME);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    ಧ್ರುವ size_t string_size = apir_decode_array_size_unchecked(decoder);
    ವರ್ಣ *       string      = (ವರ್ಣ *) apir_decoder_alloc_array(ಪರಿಮಾಣ(ವರ್ಣ), string_size);
    ಯದಿ (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }
    apir_decode_char_array(decoder, string, string_size);

    remote_call_finish(gpu, encoder, decoder);

    ಪ್ರತ್ಯಾಗಮ string;
}

ವರ್ಣ * apir_device_get_description(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ಧ್ರುವ size_t string_size = apir_decode_array_size_unchecked(decoder);
    ವರ್ಣ *       string      = (ವರ್ಣ *) apir_decoder_alloc_array(ಪರಿಮಾಣ(ವರ್ಣ), string_size);
    ಯದಿ (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device description buffer\n", __func__);

        ಪ್ರತ್ಯಾಗಮ ರಿಕ್ತ;
    }
    apir_decode_char_array(decoder, string, string_size);

    remote_call_finish(gpu, encoder, decoder);

    ಪ್ರತ್ಯಾಗಮ string;
}

uint32_t apir_device_get_type(virtgpu * gpu) {
    ಸ್ಥಿರ uint32_t dev_type = 255;
    ಯದಿ (dev_type != 255) {
        ಪ್ರತ್ಯಾಗಮ dev_type;
    }

    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_uint32_t(decoder, &dev_type);

    remote_call_finish(gpu, encoder, decoder);

    ಪ್ರತ್ಯಾಗಮ dev_type;
}

ಶೂನ್ಯ apir_device_get_memory(virtgpu * gpu, size_t * ಮುಕ್ತ, size_t * total) {
    ಸ್ಥಿರ size_t         dev_free  = 0;
    ಸ್ಥಿರ size_t         dev_total = 0;
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_MEMORY);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_size_t(decoder, &dev_free);
    apir_decode_size_t(decoder, &dev_total);

    *ಮುಕ್ತ  = dev_free;
    *total = dev_total;

    remote_call_finish(gpu, encoder, decoder);

    ಪ್ರತ್ಯಾಗಮ;
}

ತಾರ್ಕಿಕ apir_device_supports_op(virtgpu * gpu, ಧ್ರುವ ggml_tensor * op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ತಾರ್ಕಿಕ supports_op;
    apir_decode_bool_t(decoder, &supports_op);

    remote_call_finish(gpu, encoder, decoder);

    ಪ್ರತ್ಯಾಗಮ supports_op;
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

    ಪ್ರತ್ಯಾಗಮ buft_handle;
}

ಶೂನ್ಯ apir_device_get_props(virtgpu * gpu,
                           ತಾರ್ಕಿಕ *    async,
                           ತಾರ್ಕಿಕ *    host_buffer,
                           ತಾರ್ಕಿಕ *    buffer_from_host_ptr,
                           ತಾರ್ಕಿಕ *    events,
                           ತಾರ್ಕಿಕ *    mmap_support) {
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

    ಪ್ರತ್ಯಾಗಮ;
}

apir_buffer_context_t apir_device_buffer_from_ptr(virtgpu * gpu, size_t size, size_t max_tensor_size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    apir_buffer_context_t buffer_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR);

    ಯದಿ (virtgpu_shmem_create(gpu, size, &buffer_context.shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate %ldb of guest-host shared buffer", __func__, size);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, buffer_context.shmem.res_id);

    apir_encode_size_t(encoder, &size);
    apir_encode_size_t(encoder, &max_tensor_size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_buffer_host_handle_t(decoder, &buffer_context.host_handle);
    buffer_context.buft_host_handle = apir_decode_apir_buffer_type_host_handle(decoder);

    remote_call_finish(gpu, encoder, decoder);

    ಪ್ರತ್ಯಾಗಮ buffer_context;
}
