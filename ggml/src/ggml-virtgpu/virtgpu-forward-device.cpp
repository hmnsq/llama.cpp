#ಅನ್ತರ್ಭಾವ "virtgpu-forward-impl.h"
#ಅನ್ತರ್ಭಾವ "virtgpu-shm.h"

ಪೂರ್ಣಾಂಕ apir_device_get_ಎಣಿಕೆ(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_ಎಣಿಕೆ);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    int32_t dev_ಎಣಿಕೆ = -1;
    apir_decode_int32_t(decoder, &dev_ಎಣಿಕೆ);

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    ಹಿಂತಿರುಗಿ dev_ಎಣಿಕೆ;
}

ಅಕ್ಷರ * apir_device_get_name(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_NAME);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    ಸ್ಥಿರಾಂಕ size_t string_ಗಾತ್ರ = apir_decode_array_ಗಾತ್ರ_unchecked(decoder);
    ಅಕ್ಷರ *       string      = (ಅಕ್ಷರ *) apir_decoder_alloc_array(ಗಾತ್ರ(ಅಕ್ಷರ), string_ಗಾತ್ರ);
    ಆದರೆ (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }
    apir_decode_char_array(decoder, string, string_ಗಾತ್ರ);

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    ಹಿಂತಿರುಗಿ string;
}

ಅಕ್ಷರ * apir_device_get_description(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ಸ್ಥಿರಾಂಕ size_t string_ಗಾತ್ರ = apir_decode_array_ಗಾತ್ರ_unchecked(decoder);
    ಅಕ್ಷರ *       string      = (ಅಕ್ಷರ *) apir_decoder_alloc_array(ಗಾತ್ರ(ಅಕ್ಷರ), string_ಗಾತ್ರ);
    ಆದರೆ (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device description buffer\n", __func__);

        ಹಿಂತಿರುಗಿ ಖಾಲಿ;
    }
    apir_decode_char_array(decoder, string, string_ಗಾತ್ರ);

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    ಹಿಂತಿರುಗಿ string;
}

uint32_t apir_device_get_type(virtgpu * gpu) {
    ಸ್ಥಿರ uint32_t dev_type = 255;
    ಆದರೆ (dev_type != 255) {
        ಹಿಂತಿರುಗಿ dev_type;
    }

    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_uint32_t(decoder, &dev_type);

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    ಹಿಂತಿರುಗಿ dev_type;
}

ಶೂನ್ಯ apir_device_get_ಸ್ಮರಣೆ(virtgpu * gpu, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
    ಸ್ಥಿರ size_t         dev_free  = 0;
    ಸ್ಥಿರ size_t         dev_ಒಟ್ಟು = 0;
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_ಸ್ಮರಣೆ);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_ಗಾತ್ರ_t(decoder, &dev_free);
    apir_decode_ಗಾತ್ರ_t(decoder, &dev_ಒಟ್ಟು);

    *ಬಿಡುಗಡೆ  = dev_free;
    *ಒಟ್ಟು = dev_ಒಟ್ಟು;

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    ಹಿಂತಿರುಗಿ;
}

ತಾರ್ಕಿಕ apir_device_supports_op(virtgpu * gpu, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ತಾರ್ಕಿಕ supports_op;
    apir_decode_bool_t(decoder, &supports_op);

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    ಹಿಂತಿರುಗಿ supports_op;
}

apir_ಬಫರ್_type_host_handle_t apir_device_get_ಬಫರ್_type(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_ಬಫರ್_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_ಬಫರ್_type_host_handle_t buft_handle;
    apir_decode_apir_ಬಫರ್_type_host_handle_t(decoder, &buft_handle);

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    ಹಿಂತಿರುಗಿ buft_handle;
}

ಶೂನ್ಯ apir_device_get_props(virtgpu * gpu,
                           ತಾರ್ಕಿಕ *    async,
                           ತಾರ್ಕಿಕ *    host_ಬಫರ್,
                           ತಾರ್ಕಿಕ *    ಬಫರ್_from_host_ptr,
                           ತಾರ್ಕಿಕ *    events,
                           ತಾರ್ಕಿಕ *    mmap_support) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_PROPS);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_bool_t(decoder, async);
    apir_decode_bool_t(decoder, host_ಬಫರ್);
    apir_decode_bool_t(decoder, ಬಫರ್_from_host_ptr);
    apir_decode_bool_t(decoder, events);
    apir_decode_bool_t(decoder, mmap_support);

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    ಹಿಂತಿರುಗಿ;
}

apir_ಬಫರ್_context_t apir_device_ಬಫರ್_from_ptr(virtgpu * gpu, size_t size, size_t max_tensor_ಗಾತ್ರ) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    apir_ಬಫರ್_context_t ಬಫರ್_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_ಬಫರ್_FROM_PTR);

    ಆದರೆ (virtgpu_shmem_create(gpu, size, &ಬಫರ್_context.shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate %ldb of guest-host shared buffer", __func__, size);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, ಬಫರ್_context.shmem.res_id);

    apir_encode_ಗಾತ್ರ_t(encoder, &size);
    apir_encode_ಗಾತ್ರ_t(encoder, &max_tensor_ಗಾತ್ರ);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_ಬಫರ್_host_handle_t(decoder, &ಬಫರ್_context.host_handle);
    ಬಫರ್_context.buft_host_handle = apir_decode_apir_ಬಫರ್_type_host_handle(decoder);

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    ಹಿಂತಿರುಗಿ ಬಫರ್_context;
}
