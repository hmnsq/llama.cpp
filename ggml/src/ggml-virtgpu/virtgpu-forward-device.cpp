#ضمن "virtgpu-forward-impl.h"
#ضمن "virtgpu-shm.h"

صحيح apir_device_get_شمار(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_شمار);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    int32_t dev_شمار = -1;
    apir_decode_int32_t(decoder, &dev_شمار);

    remote_call_پایان(gpu, encoder, decoder);

    أرجع dev_شمار;
}

حرف * apir_device_get_name(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_NAME);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    ثابت size_t string_اندازه = apir_decode_array_اندازه_unchecked(decoder);
    حرف *       string      = (حرف *) apir_decoder_alloc_array(حجم(حرف), string_اندازه);
    إذا (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        أرجع عدم;
    }
    apir_decode_char_array(decoder, string, string_اندازه);

    remote_call_پایان(gpu, encoder, decoder);

    أرجع string;
}

حرف * apir_device_get_description(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ثابت size_t string_اندازه = apir_decode_array_اندازه_unchecked(decoder);
    حرف *       string      = (حرف *) apir_decoder_alloc_array(حجم(حرف), string_اندازه);
    إذا (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device description buffer\n", __func__);

        أرجع عدم;
    }
    apir_decode_char_array(decoder, string, string_اندازه);

    remote_call_پایان(gpu, encoder, decoder);

    أرجع string;
}

uint32_t apir_device_get_type(virtgpu * gpu) {
    ساكن uint32_t dev_type = 255;
    إذا (dev_type != 255) {
        أرجع dev_type;
    }

    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_uint32_t(decoder, &dev_type);

    remote_call_پایان(gpu, encoder, decoder);

    أرجع dev_type;
}

فراغ apir_device_get_حافظه(virtgpu * gpu, size_t * حرر, size_t * مجموع) {
    ساكن size_t         dev_free  = 0;
    ساكن size_t         dev_مجموع = 0;
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_حافظه);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_اندازه_t(decoder, &dev_free);
    apir_decode_اندازه_t(decoder, &dev_مجموع);

    *حرر  = dev_free;
    *مجموع = dev_مجموع;

    remote_call_پایان(gpu, encoder, decoder);

    أرجع;
}

منطقي apir_device_supports_op(virtgpu * gpu, ثابت ggml_tensor * op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    منطقي supports_op;
    apir_decode_bool_t(decoder, &supports_op);

    remote_call_پایان(gpu, encoder, decoder);

    أرجع supports_op;
}

apir_میانگیر_type_host_handle_t apir_device_get_میانگیر_type(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_میانگیر_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_میانگیر_type_host_handle_t buft_handle;
    apir_decode_apir_میانگیر_type_host_handle_t(decoder, &buft_handle);

    remote_call_پایان(gpu, encoder, decoder);

    أرجع buft_handle;
}

فراغ apir_device_get_props(virtgpu * gpu,
                           منطقي *    async,
                           منطقي *    host_میانگیر,
                           منطقي *    میانگیر_from_host_ptr,
                           منطقي *    events,
                           منطقي *    mmap_support) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_PROPS);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_bool_t(decoder, async);
    apir_decode_bool_t(decoder, host_میانگیر);
    apir_decode_bool_t(decoder, میانگیر_from_host_ptr);
    apir_decode_bool_t(decoder, events);
    apir_decode_bool_t(decoder, mmap_support);

    remote_call_پایان(gpu, encoder, decoder);

    أرجع;
}

apir_میانگیر_context_t apir_device_میانگیر_from_ptr(virtgpu * gpu, size_t size, size_t max_tensor_اندازه) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    apir_میانگیر_context_t میانگیر_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_میانگیر_FROM_PTR);

    إذا (virtgpu_shmem_create(gpu, size, &میانگیر_context.shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate %ldb of guest-host shared buffer", __func__, size);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, میانگیر_context.shmem.res_id);

    apir_encode_اندازه_t(encoder, &size);
    apir_encode_اندازه_t(encoder, &max_tensor_اندازه);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_میانگیر_host_handle_t(decoder, &میانگیر_context.host_handle);
    میانگیر_context.buft_host_handle = apir_decode_apir_میانگیر_type_host_handle(decoder);

    remote_call_پایان(gpu, encoder, decoder);

    أرجع میانگیر_context;
}
