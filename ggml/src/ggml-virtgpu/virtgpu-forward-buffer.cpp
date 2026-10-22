#समावेश "virtgpu-forward-impl.h"

व्योम * apir_बफर_get_base(virtgpu * gpu, apir_बफर_context_t * बफर_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_GET_BASE);

    apir_encode_apir_बफर_host_handle_t(encoder, &बफर_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    uintptr_t base;
    apir_decode_uintptr_t(decoder, &base);

    remote_call_finish(gpu, encoder, decoder);

    वापस (व्योम *) base;
}

व्योम apir_बफर_set_tensor(virtgpu *               gpu,
                            apir_बफर_context_t * बफर_context,
                            ggml_tensor *           tensor,
                            स्थिर व्योम *            तथ्य,
                            माप_प्रकार                  offset,
                            माप_प्रकार                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_SET_TENSOR);

    apir_encode_apir_बफर_host_handle_t(encoder, &बफर_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   अस्थायी_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &अस्थायी_shmem;
    bool            using_shared_shmem = false;

    अगर (size <= gpu->तथ्य_shmem.mmap_आकार) {
        // Lock mutex before using shared data_shmem buffer
        अगर (mtx_lock(&gpu->तथ्य_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = true;
        shmem              = &gpu->तथ्य_shmem;

    } अन्यथा अगर (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    स_नकल(shmem->mmap_ptr, तथ्य, size);
    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_आकार_t(encoder, &offset);
    apir_encode_आकार_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_finish(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    अगर (using_shared_shmem) {
        mtx_unlock(&gpu->तथ्य_shmem_mutex);
    } अन्यथा {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    वापस;
}

व्योम apir_बफर_get_tensor(virtgpu *               gpu,
                            apir_बफर_context_t * बफर_context,
                            स्थिर ggml_tensor *     tensor,
                            व्योम *                  तथ्य,
                            माप_प्रकार                  offset,
                            माप_प्रकार                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_GET_TENSOR);

    apir_encode_apir_बफर_host_handle_t(encoder, &बफर_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   अस्थायी_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &अस्थायी_shmem;
    bool            using_shared_shmem = false;

    अगर (size <= gpu->तथ्य_shmem.mmap_आकार) {
        // Lock mutex before using shared data_shmem buffer
        अगर (mtx_lock(&gpu->तथ्य_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = true;
        shmem              = &gpu->तथ्य_shmem;

    } अन्यथा अगर (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);
    apir_encode_आकार_t(encoder, &offset);
    apir_encode_आकार_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    स_नकल(तथ्य, shmem->mmap_ptr, size);

    remote_call_finish(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    अगर (using_shared_shmem) {
        mtx_unlock(&gpu->तथ्य_shmem_mutex);
    } अन्यथा {
        virtgpu_shmem_destroy(gpu, shmem);
    }
}

bool apir_बफर_cpy_tensor(virtgpu *               gpu,
                            apir_बफर_context_t * बफर_context,
                            स्थिर ggml_tensor *     src,
                            स्थिर ggml_tensor *     dst) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_CPY_TENSOR);

    apir_encode_apir_बफर_host_handle_t(encoder, &बफर_context->host_handle);
    apir_encode_ggml_tensor(encoder, src);
    apir_encode_ggml_tensor(encoder, dst);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    bool ret_val;
    apir_decode_bool_t(decoder, &ret_val);

    remote_call_finish(gpu, encoder, decoder);

    वापस ret_val;
}

व्योम apir_बफर_clear(virtgpu * gpu, apir_बफर_context_t * बफर_context, uint8_t value) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_CLEAR);

    apir_encode_apir_बफर_host_handle_t(encoder, &बफर_context->host_handle);
    apir_encode_uint8_t(encoder, &value);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_finish(gpu, encoder, decoder);
}

व्योम apir_बफर_free_बफर(virtgpu * gpu, apir_बफर_context_t * बफर_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_FREE_बफर);

    apir_encode_apir_बफर_host_handle_t(encoder, &बफर_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_finish(gpu, encoder, decoder);
}
