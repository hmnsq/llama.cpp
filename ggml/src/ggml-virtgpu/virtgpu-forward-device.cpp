#অন্তর্ভাৱ "virtgpu-forward-impl.h"
#অন্তর্ভাৱ "virtgpu-shm.h"

পূর্ণাঙ্ক apir_device_get_গুনতি(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_গুনতি);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    int32_t dev_গুনতি = -1;
    apir_decode_int32_t(decoder, &dev_গুনতি);

    remote_call_finish(gpu, encoder, decoder);

    ঘূৰাই dev_গুনতি;
}

ৱর্ণ * apir_device_get_name(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_NAME);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    ধ্রুৱ size_t string_size = apir_decode_array_size_unchecked(decoder);
    ৱর্ণ *       string      = (ৱর্ণ *) apir_decoder_alloc_array(পরিমাণ(ৱর্ণ), string_size);
    যদি (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        ঘূৰাই রিক্ত;
    }
    apir_decode_char_array(decoder, string, string_size);

    remote_call_finish(gpu, encoder, decoder);

    ঘূৰাই string;
}

ৱর্ণ * apir_device_get_description(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ধ্রুৱ size_t string_size = apir_decode_array_size_unchecked(decoder);
    ৱর্ণ *       string      = (ৱর্ণ *) apir_decoder_alloc_array(পরিমাণ(ৱর্ণ), string_size);
    যদি (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device description buffer\n", __func__);

        ঘূৰাই রিক্ত;
    }
    apir_decode_char_array(decoder, string, string_size);

    remote_call_finish(gpu, encoder, decoder);

    ঘূৰাই string;
}

uint32_t apir_device_get_type(virtgpu * gpu) {
    স্থির uint32_t dev_type = 255;
    যদি (dev_type != 255) {
        ঘূৰাই dev_type;
    }

    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_uint32_t(decoder, &dev_type);

    remote_call_finish(gpu, encoder, decoder);

    ঘূৰাই dev_type;
}

শূন্য apir_device_get_memory(virtgpu * gpu, size_t * মুক্ত, size_t * total) {
    স্থির size_t         dev_free  = 0;
    স্থির size_t         dev_total = 0;
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_MEMORY);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_size_t(decoder, &dev_free);
    apir_decode_size_t(decoder, &dev_total);

    *মুক্ত  = dev_free;
    *total = dev_total;

    remote_call_finish(gpu, encoder, decoder);

    ঘূৰাই;
}

তার্কিক apir_device_supports_op(virtgpu * gpu, ধ্রুৱ ggml_tensor * op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    তার্কিক supports_op;
    apir_decode_bool_t(decoder, &supports_op);

    remote_call_finish(gpu, encoder, decoder);

    ঘূৰাই supports_op;
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

    ঘূৰাই buft_handle;
}

শূন্য apir_device_get_props(virtgpu * gpu,
                           তার্কিক *    async,
                           তার্কিক *    host_buffer,
                           তার্কিক *    buffer_from_host_ptr,
                           তার্কিক *    events,
                           তার্কিক *    mmap_support) {
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

    ঘূৰাই;
}

apir_buffer_context_t apir_device_buffer_from_ptr(virtgpu * gpu, size_t size, size_t max_tensor_size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    apir_buffer_context_t buffer_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_BUFFER_FROM_PTR);

    যদি (virtgpu_shmem_create(gpu, size, &buffer_context.shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate %ldb of guest-host shared buffer", __func__, size);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, buffer_context.shmem.res_id);

    apir_encode_size_t(encoder, &size);
    apir_encode_size_t(encoder, &max_tensor_size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_buffer_host_handle_t(decoder, &buffer_context.host_handle);
    buffer_context.buft_host_handle = apir_decode_apir_buffer_type_host_handle(decoder);

    remote_call_finish(gpu, encoder, decoder);

    ঘূৰাই buffer_context;
}
