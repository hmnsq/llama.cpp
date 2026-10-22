#అన్తర్భావ "virtgpu-forward-impl.h"
#అన్తర్భావ "virtgpu-shm.h"

పూర్ణాంకం apir_device_get_లెక్క(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_లెక్క);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    int32_t dev_లెక్క = -1;
    apir_decode_int32_t(decoder, &dev_లెక్క);

    remote_call_ముగింపు(gpu, encoder, decoder);

    తిరిగి dev_లెక్క;
}

అక్షరం * apir_device_get_name(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_NAME);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    స్థిరరాశి size_t string_కొలత = apir_decode_array_కొలత_unchecked(decoder);
    అక్షరం *       string      = (అక్షరం *) apir_decoder_alloc_array(పరిమాణం(అక్షరం), string_కొలత);
    ఒకవేళ (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        తిరిగి ఖాళీ;
    }
    apir_decode_char_array(decoder, string, string_కొలత);

    remote_call_ముగింపు(gpu, encoder, decoder);

    తిరిగి string;
}

అక్షరం * apir_device_get_description(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    స్థిరరాశి size_t string_కొలత = apir_decode_array_కొలత_unchecked(decoder);
    అక్షరం *       string      = (అక్షరం *) apir_decoder_alloc_array(పరిమాణం(అక్షరం), string_కొలత);
    ఒకవేళ (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device description buffer\n", __func__);

        తిరిగి ఖాళీ;
    }
    apir_decode_char_array(decoder, string, string_కొలత);

    remote_call_ముగింపు(gpu, encoder, decoder);

    తిరిగి string;
}

uint32_t apir_device_get_type(virtgpu * gpu) {
    స్థిర uint32_t dev_type = 255;
    ఒకవేళ (dev_type != 255) {
        తిరిగి dev_type;
    }

    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_uint32_t(decoder, &dev_type);

    remote_call_ముగింపు(gpu, encoder, decoder);

    తిరిగి dev_type;
}

శూన్య apir_device_get_జ్ఞాపకం(virtgpu * gpu, size_t * విడుదల, size_t * మొత్తం) {
    స్థిర size_t         dev_free  = 0;
    స్థిర size_t         dev_మొత్తం = 0;
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_జ్ఞాపకం);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_కొలత_t(decoder, &dev_free);
    apir_decode_కొలత_t(decoder, &dev_మొత్తం);

    *విడుదల  = dev_free;
    *మొత్తం = dev_మొత్తం;

    remote_call_ముగింపు(gpu, encoder, decoder);

    తిరిగి;
}

తార్కిక apir_device_supports_op(virtgpu * gpu, స్థిరరాశి ggml_tensor * op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    తార్కిక supports_op;
    apir_decode_bool_t(decoder, &supports_op);

    remote_call_ముగింపు(gpu, encoder, decoder);

    తిరిగి supports_op;
}

apir_బఫరు_type_host_handle_t apir_device_get_బఫరు_type(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_బఫరు_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_బఫరు_type_host_handle_t buft_handle;
    apir_decode_apir_బఫరు_type_host_handle_t(decoder, &buft_handle);

    remote_call_ముగింపు(gpu, encoder, decoder);

    తిరిగి buft_handle;
}

శూన్య apir_device_get_props(virtgpu * gpu,
                           తార్కిక *    async,
                           తార్కిక *    host_బఫరు,
                           తార్కిక *    బఫరు_from_host_ptr,
                           తార్కిక *    events,
                           తార్కిక *    mmap_support) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_PROPS);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_bool_t(decoder, async);
    apir_decode_bool_t(decoder, host_బఫరు);
    apir_decode_bool_t(decoder, బఫరు_from_host_ptr);
    apir_decode_bool_t(decoder, events);
    apir_decode_bool_t(decoder, mmap_support);

    remote_call_ముగింపు(gpu, encoder, decoder);

    తిరిగి;
}

apir_బఫరు_context_t apir_device_బఫరు_from_ptr(virtgpu * gpu, size_t size, size_t max_tensor_కొలత) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    apir_బఫరు_context_t బఫరు_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_బఫరు_FROM_PTR);

    ఒకవేళ (virtgpu_shmem_create(gpu, size, &బఫరు_context.shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate %ldb of guest-host shared buffer", __func__, size);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, బఫరు_context.shmem.res_id);

    apir_encode_కొలత_t(encoder, &size);
    apir_encode_కొలత_t(encoder, &max_tensor_కొలత);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_బఫరు_host_handle_t(decoder, &బఫరు_context.host_handle);
    బఫరు_context.buft_host_handle = apir_decode_apir_బఫరు_type_host_handle(decoder);

    remote_call_ముగింపు(gpu, encoder, decoder);

    తిరిగి బఫరు_context;
}
