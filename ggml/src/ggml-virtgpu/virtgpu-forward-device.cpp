#অন্তর্ভাব "virtgpu-forward-impl.h"
#অন্তর্ভাব "virtgpu-shm.h"

পূর্ণসংখ্যা apir_device_get_গুনতি(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_গুনতি);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    int32_t dev_গুনতি = -1;
    apir_decode_int32_t(decoder, &dev_গুনতি);

    remote_call_finish(gpu, encoder, decoder);

    ফেরত dev_গুনতি;
}

অক্ষর * apir_device_get_name(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_NAME);
    REMOTE_CALL(gpu, encoder, decoder, ret);

    ধ্রুবক size_t string_আকার = apir_decode_array_আকার_unchecked(decoder);
    অক্ষর *       string      = (অক্ষর *) apir_decoder_alloc_array(আকার(অক্ষর), string_আকার);
    যদি (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device name buffer\n", __func__);
        ফেরত ফাঁকা;
    }
    apir_decode_char_array(decoder, string, string_আকার);

    remote_call_finish(gpu, encoder, decoder);

    ফেরত string;
}

অক্ষর * apir_device_get_description(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_DESCRIPTION);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ধ্রুবক size_t string_আকার = apir_decode_array_আকার_unchecked(decoder);
    অক্ষর *       string      = (অক্ষর *) apir_decoder_alloc_array(আকার(অক্ষর), string_আকার);
    যদি (!string) {
        GGML_LOG_ERROR(GGML_VIRTGPU "%s: Could not allocate the device description buffer\n", __func__);

        ফেরত ফাঁকা;
    }
    apir_decode_char_array(decoder, string, string_আকার);

    remote_call_finish(gpu, encoder, decoder);

    ফেরত string;
}

uint32_t apir_device_get_type(virtgpu * gpu) {
    স্থির uint32_t dev_type = 255;
    যদি (dev_type != 255) {
        ফেরত dev_type;
    }

    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_uint32_t(decoder, &dev_type);

    remote_call_finish(gpu, encoder, decoder);

    ফেরত dev_type;
}

শূন্য apir_device_get_memory(virtgpu * gpu, size_t * মুক্ত, size_t * যোগ) {
    স্থির size_t         dev_free  = 0;
    স্থির size_t         dev_যোগ = 0;
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_MEMORY);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_আকার_t(decoder, &dev_free);
    apir_decode_আকার_t(decoder, &dev_যোগ);

    *মুক্ত  = dev_free;
    *যোগ = dev_যোগ;

    remote_call_finish(gpu, encoder, decoder);

    ফেরত;
}

যৌক্তিক apir_device_supports_op(virtgpu * gpu, ধ্রুবক ggml_tensor * op) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_SUPPORTS_OP);

    apir_encode_ggml_tensor_inline(encoder, op);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    যৌক্তিক supports_op;
    apir_decode_bool_t(decoder, &supports_op);

    remote_call_finish(gpu, encoder, decoder);

    ফেরত supports_op;
}

apir_বাফার_type_host_handle_t apir_device_get_বাফার_type(virtgpu * gpu) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_বাফার_TYPE);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_বাফার_type_host_handle_t buft_handle;
    apir_decode_apir_বাফার_type_host_handle_t(decoder, &buft_handle);

    remote_call_finish(gpu, encoder, decoder);

    ফেরত buft_handle;
}

শূন্য apir_device_get_props(virtgpu * gpu,
                           যৌক্তিক *    async,
                           যৌক্তিক *    host_বাফার,
                           যৌক্তিক *    বাফার_from_host_ptr,
                           যৌক্তিক *    events,
                           যৌক্তিক *    mmap_support) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_GET_PROPS);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_bool_t(decoder, async);
    apir_decode_bool_t(decoder, host_বাফার);
    apir_decode_bool_t(decoder, বাফার_from_host_ptr);
    apir_decode_bool_t(decoder, events);
    apir_decode_bool_t(decoder, mmap_support);

    remote_call_finish(gpu, encoder, decoder);

    ফেরত;
}

apir_বাফার_context_t apir_device_বাফার_from_ptr(virtgpu * gpu, size_t size, size_t max_tensor_আকার) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    apir_বাফার_context_t বাফার_context;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_DEVICE_বাফার_FROM_PTR);

    যদি (virtgpu_shmem_create(gpu, size, &বাফার_context.shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate %ldb of guest-host shared buffer", __func__, size);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, বাফার_context.shmem.res_id);

    apir_encode_আকার_t(encoder, &size);
    apir_encode_আকার_t(encoder, &max_tensor_আকার);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    apir_decode_apir_বাফার_host_handle_t(decoder, &বাফার_context.host_handle);
    বাফার_context.buft_host_handle = apir_decode_apir_বাফার_type_host_handle(decoder);

    remote_call_finish(gpu, encoder, decoder);

    ফেরত বাফার_context;
}
