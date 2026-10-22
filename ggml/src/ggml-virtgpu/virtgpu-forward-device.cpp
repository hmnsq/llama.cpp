#અન્તર્ભાવ "virtgpu-forward-impl.h"
#અન્તર્ભાવ "virtgpu-shm.h"

પૂર્ણાંક apir_device_get_ગણતરી(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_ગણતરી);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    int32_t dev_ગણતરી = -1;
    apir_decode_int32_t(decoder, &dev_ગણતરી);

    remote_call_finish(gpu, encoder, decoder);

    પાછા dev_ગણતરી;
}

અક્ષર * apir_device_get_name(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_NAME);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    અચલ size_t string_size = apir_decode_array_size_unchecked(decoder);
    અક્ષર *       string      = (અક્ષર *) apir_decoder_alloc_array(કદ(અક્ષર), string_size);
    જો (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        પાછા ખાલી;
    }
    apir_decode_char_array(decoder, string, string_size);

    remote_call_finish(gpu, encoder, decoder);

    પાછા string;
}

અક્ષર * apir_device_get_description(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    અચલ size_t string_size = apir_decode_array_size_unchecked(decoder);
    અક્ષર *       string      = (અક્ષર *) apir_decoder_alloc_array(કદ(અક્ષર), string_size);
    જો (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device description buffer\n", __func__);

        પાછા ખાલી;
    }
    apir_decode_char_array(decoder, string, string_size);

    remote_call_finish(gpu, encoder, decoder);

    પાછા string;
}

uint32_t apir_device_get_type(virtgpu * gpu) {
    સ્થિર uint32_t dev_type = 255;
    જો (dev_type != 255) {
        પાછા dev_type;
    }

    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_uint32_t(decoder, &dev_type);

    remote_call_finish(gpu, encoder, decoder);

    પાછા dev_type;
}

શૂન્ય apir_device_get_memory(virtgpu * gpu, size_t * મુક્ત, size_t * total) {
    સ્થિર size_t         dev_free  = 0;
    સ્થિર size_t         dev_total = 0;
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_MEMORY);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_size_t(decoder, &dev_free);
    apir_decode_size_t(decoder, &dev_total);

    *મુક્ત  = dev_free;
    *total = dev_total;

    remote_call_finish(gpu, encoder, decoder);

    પાછા;
}

તાર્કિક apir_device_supports_op(virtgpu * gpu, અચલ ggml_tensor * op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    તાર્કિક supports_op;
    apir_decode_bool_t(decoder, &supports_op);

    remote_call_finish(gpu, encoder, decoder);

    પાછા supports_op;
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

    પાછા buft_handle;
}

શૂન્ય apir_device_get_props(virtgpu * gpu,
                           તાર્કિક *    async,
                           તાર્કિક *    host_buffer,
                           તાર્કિક *    buffer_from_host_ptr,
                           તાર્કિક *    events,
                           તાર્કિક *    mmap_support) {
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

    પાછા;
}

apir_buffer_context_t apir_device_buffer_from_ptr(virtgpu * gpu, size_t size, size_t max_tensor_size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    apir_buffer_context_t buffer_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR);

    જો (virtgpu_shmem_create(gpu, size, &buffer_context.shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate %ldb of guest-host shared buffer", __func__, size);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, buffer_context.shmem.res_id);

    apir_encode_size_t(encoder, &size);
    apir_encode_size_t(encoder, &max_tensor_size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_buffer_host_handle_t(decoder, &buffer_context.host_handle);
    buffer_context.buft_host_handle = apir_decode_apir_buffer_type_host_handle(decoder);

    remote_call_finish(gpu, encoder, decoder);

    પાછા buffer_context;
}
