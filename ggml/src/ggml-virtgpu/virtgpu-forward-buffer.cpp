#अन्तर्भाव "virtgpu-forward-impl.h"

शून्य * apir_buffer_get_base(virtgpu * gpu, apir_buffer_context_t * buffer_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_GET_BASE);

    apir_encode_apir_buffer_host_handle_t(encoder, &buffer_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    uintptr_t base;
    apir_decode_uintptr_t(decoder, &base);

    remote_call_finish(gpu, encoder, decoder);

    परत (शून्य *) base;
}

शून्य apir_buffer_set_tensor(virtgpu *               gpu,
                            apir_buffer_context_t * buffer_context,
                            ggml_tensor *           tensor,
                            स्थिरांक शून्य *            data,
                            माप_प्रकार                  offset,
                            माप_प्रकार                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_SET_TENSOR);

    apir_encode_apir_buffer_host_handle_t(encoder, &buffer_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   temp_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &temp_shmem;
    तार्किक            using_shared_shmem = असत्य;

    जर (size <= gpu->data_shmem.mmap_size) {
        // Lock mutex before using shared data_shmem buffer
        जर (mtx_lock(&gpu->data_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = सत्य;
        shmem              = &gpu->data_shmem;

    } अन्यथा जर (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    स_नकल(shmem->mmap_ptr, data, size);
    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_size_t(encoder, &offset);
    apir_encode_size_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_finish(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    जर (using_shared_shmem) {
        mtx_unlock(&gpu->data_shmem_mutex);
    } अन्यथा {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    परत;
}

शून्य apir_buffer_get_tensor(virtgpu *               gpu,
                            apir_buffer_context_t * buffer_context,
                            स्थिरांक ggml_tensor *     tensor,
                            शून्य *                  data,
                            माप_प्रकार                  offset,
                            माप_प्रकार                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_GET_TENSOR);

    apir_encode_apir_buffer_host_handle_t(encoder, &buffer_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   temp_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &temp_shmem;
    तार्किक            using_shared_shmem = असत्य;

    जर (size <= gpu->data_shmem.mmap_size) {
        // Lock mutex before using shared data_shmem buffer
        जर (mtx_lock(&gpu->data_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = सत्य;
        shmem              = &gpu->data_shmem;

    } अन्यथा जर (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);
    apir_encode_size_t(encoder, &offset);
    apir_encode_size_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    स_नकल(data, shmem->mmap_ptr, size);

    remote_call_finish(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    जर (using_shared_shmem) {
        mtx_unlock(&gpu->data_shmem_mutex);
    } अन्यथा {
        virtgpu_shmem_destroy(gpu, shmem);
    }
}

तार्किक apir_buffer_cpy_tensor(virtgpu *               gpu,
                            apir_buffer_context_t * buffer_context,
                            स्थिरांक ggml_tensor *     src,
                            स्थिरांक ggml_tensor *     dst) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_CPY_TENSOR);

    apir_encode_apir_buffer_host_handle_t(encoder, &buffer_context->host_handle);
    apir_encode_ggml_tensor(encoder, src);
    apir_encode_ggml_tensor(encoder, dst);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    तार्किक ret_val;
    apir_decode_bool_t(decoder, &ret_val);

    remote_call_finish(gpu, encoder, decoder);

    परत ret_val;
}

शून्य apir_buffer_clear(virtgpu * gpu, apir_buffer_context_t * buffer_context, uint8_t value) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_CLEAR);

    apir_encode_apir_buffer_host_handle_t(encoder, &buffer_context->host_handle);
    apir_encode_uint8_t(encoder, &value);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_finish(gpu, encoder, decoder);
}

शून्य apir_buffer_free_buffer(virtgpu * gpu, apir_buffer_context_t * buffer_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BUFFER_FREE_BUFFER);

    apir_encode_apir_buffer_host_handle_t(encoder, &buffer_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_finish(gpu, encoder, decoder);
}
