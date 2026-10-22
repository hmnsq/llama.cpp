#অন্তর্ভাব "virtgpu-forward-impl.h"

শূন্য * apir_বাফার_get_base(virtgpu * gpu, apir_বাফার_context_t * বাফার_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_বাফার_GET_BASE);

    apir_encode_apir_বাফার_host_handle_t(encoder, &বাফার_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    uintptr_t base;
    apir_decode_uintptr_t(decoder, &base);

    remote_call_finish(gpu, encoder, decoder);

    ফেরত (শূন্য *) base;
}

শূন্য apir_বাফার_set_tensor(virtgpu *               gpu,
                            apir_বাফার_context_t * বাফার_context,
                            ggml_tensor *           tensor,
                            ধ্রুবক শূন্য *            তথ্য,
                            size_t                  offset,
                            size_t                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_বাফার_SET_TENSOR);

    apir_encode_apir_বাফার_host_handle_t(encoder, &বাফার_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   অস্থায়ী_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &অস্থায়ী_shmem;
    যৌক্তিক            using_shared_shmem = মিথ্যা;

    যদি (size <= gpu->তথ্য_shmem.mmap_আকার) {
        // Lock mutex before using shared data_shmem buffer
        যদি (mtx_lock(&gpu->তথ্য_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = সত্য;
        shmem              = &gpu->তথ্য_shmem;

    } অন্যথা যদি (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    memcpy(shmem->mmap_ptr, তথ্য, size);
    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_আকার_t(encoder, &offset);
    apir_encode_আকার_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_finish(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    যদি (using_shared_shmem) {
        mtx_unlock(&gpu->তথ্য_shmem_mutex);
    } অন্যথা {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    ফেরত;
}

শূন্য apir_বাফার_get_tensor(virtgpu *               gpu,
                            apir_বাফার_context_t * বাফার_context,
                            ধ্রুবক ggml_tensor *     tensor,
                            শূন্য *                  তথ্য,
                            size_t                  offset,
                            size_t                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_বাফার_GET_TENSOR);

    apir_encode_apir_বাফার_host_handle_t(encoder, &বাফার_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   অস্থায়ী_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &অস্থায়ী_shmem;
    যৌক্তিক            using_shared_shmem = মিথ্যা;

    যদি (size <= gpu->তথ্য_shmem.mmap_আকার) {
        // Lock mutex before using shared data_shmem buffer
        যদি (mtx_lock(&gpu->তথ্য_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = সত্য;
        shmem              = &gpu->তথ্য_shmem;

    } অন্যথা যদি (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);
    apir_encode_আকার_t(encoder, &offset);
    apir_encode_আকার_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    memcpy(তথ্য, shmem->mmap_ptr, size);

    remote_call_finish(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    যদি (using_shared_shmem) {
        mtx_unlock(&gpu->তথ্য_shmem_mutex);
    } অন্যথা {
        virtgpu_shmem_destroy(gpu, shmem);
    }
}

যৌক্তিক apir_বাফার_cpy_tensor(virtgpu *               gpu,
                            apir_বাফার_context_t * বাফার_context,
                            ধ্রুবক ggml_tensor *     src,
                            ধ্রুবক ggml_tensor *     dst) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_বাফার_CPY_TENSOR);

    apir_encode_apir_বাফার_host_handle_t(encoder, &বাফার_context->host_handle);
    apir_encode_ggml_tensor(encoder, src);
    apir_encode_ggml_tensor(encoder, dst);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    যৌক্তিক ret_val;
    apir_decode_bool_t(decoder, &ret_val);

    remote_call_finish(gpu, encoder, decoder);

    ফেরত ret_val;
}

শূন্য apir_বাফার_clear(virtgpu * gpu, apir_বাফার_context_t * বাফার_context, uint8_t value) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_বাফার_CLEAR);

    apir_encode_apir_বাফার_host_handle_t(encoder, &বাফার_context->host_handle);
    apir_encode_uint8_t(encoder, &value);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_finish(gpu, encoder, decoder);
}

শূন্য apir_বাফার_free_বাফার(virtgpu * gpu, apir_বাফার_context_t * বাফার_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_বাফার_FREE_বাফার);

    apir_encode_apir_বাফার_host_handle_t(encoder, &বাফার_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_finish(gpu, encoder, decoder);
}
