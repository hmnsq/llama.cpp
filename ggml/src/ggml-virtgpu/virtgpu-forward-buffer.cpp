#ਅਨ੍ਤਰ੍ਭਾਵ "virtgpu-forward-impl.h"

ਖਾਲੀ * apir_ਬਫਰ_get_ਅਧਾਰ(virtgpu * gpu, apir_ਬਫਰ_context_t * ਬਫਰ_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ਬਫਰ_GET_ਅਧਾਰ);

    apir_encode_apir_ਬਫਰ_host_handle_t(encoder, &ਬਫਰ_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    uintptr_t ਅਧਾਰ;
    apir_decode_uintptr_t(decoder, &ਅਧਾਰ);

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    ਵਾਪਸ (ਖਾਲੀ *) ਅਧਾਰ;
}

ਖਾਲੀ apir_ਬਫਰ_ਸੈੱਟ_tensor(virtgpu *               gpu,
                            apir_ਬਫਰ_context_t * ਬਫਰ_context,
                            ggml_tensor *           tensor,
                            ਅਚਲ ਖਾਲੀ *            ਅੰਕੜੇ,
                            size_t                  ਵਿੱਥ,
                            size_t                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ਬਫਰ_ਸੈੱਟ_TENSOR);

    apir_encode_apir_ਬਫਰ_host_handle_t(encoder, &ਬਫਰ_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   ਆਰਜ਼ੀ_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &ਆਰਜ਼ੀ_shmem;
    ਤਾਰਕਿਕ            using_shared_shmem = ਝੂਠ;

    ਜੇ (size <= gpu->ਅੰਕੜੇ_shmem.mmap_ਮਾਪ) {
        // Lock mutex before using shared data_shmem buffer
        ਜੇ (mtx_ਤਾਲਾ(&gpu->ਅੰਕੜੇ_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = ਸੱਚ;
        shmem              = &gpu->ਅੰਕੜੇ_shmem;

    } ਨਹੀਂਤਾਂ ਜੇ (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(shmem->mmap_ptr, ਅੰਕੜੇ, size);
    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_ਮਾਪ_t(encoder, &ਵਿੱਥ);
    apir_encode_ਮਾਪ_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    ਜੇ (using_shared_shmem) {
        mtx_unlock(&gpu->ਅੰਕੜੇ_shmem_mutex);
    } ਨਹੀਂਤਾਂ {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    ਵਾਪਸ;
}

ਖਾਲੀ apir_ਬਫਰ_get_tensor(virtgpu *               gpu,
                            apir_ਬਫਰ_context_t * ਬਫਰ_context,
                            ਅਚਲ ggml_tensor *     tensor,
                            ਖਾਲੀ *                  ਅੰਕੜੇ,
                            size_t                  ਵਿੱਥ,
                            size_t                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ਬਫਰ_GET_TENSOR);

    apir_encode_apir_ਬਫਰ_host_handle_t(encoder, &ਬਫਰ_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   ਆਰਜ਼ੀ_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &ਆਰਜ਼ੀ_shmem;
    ਤਾਰਕਿਕ            using_shared_shmem = ਝੂਠ;

    ਜੇ (size <= gpu->ਅੰਕੜੇ_shmem.mmap_ਮਾਪ) {
        // Lock mutex before using shared data_shmem buffer
        ਜੇ (mtx_ਤਾਲਾ(&gpu->ਅੰਕੜੇ_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = ਸੱਚ;
        shmem              = &gpu->ਅੰਕੜੇ_shmem;

    } ਨਹੀਂਤਾਂ ਜੇ (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);
    apir_encode_ਮਾਪ_t(encoder, &ਵਿੱਥ);
    apir_encode_ਮਾਪ_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਅੰਕੜੇ, shmem->mmap_ptr, size);

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    ਜੇ (using_shared_shmem) {
        mtx_unlock(&gpu->ਅੰਕੜੇ_shmem_mutex);
    } ਨਹੀਂਤਾਂ {
        virtgpu_shmem_destroy(gpu, shmem);
    }
}

ਤਾਰਕਿਕ apir_ਬਫਰ_cpy_tensor(virtgpu *               gpu,
                            apir_ਬਫਰ_context_t * ਬਫਰ_context,
                            ਅਚਲ ggml_tensor *     src,
                            ਅਚਲ ggml_tensor *     dst) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ਬਫਰ_CPY_TENSOR);

    apir_encode_apir_ਬਫਰ_host_handle_t(encoder, &ਬਫਰ_context->host_handle);
    apir_encode_ggml_tensor(encoder, src);
    apir_encode_ggml_tensor(encoder, dst);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ਤਾਰਕਿਕ ret_val;
    apir_decode_bool_t(decoder, &ret_val);

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    ਵਾਪਸ ret_val;
}

ਖਾਲੀ apir_ਬਫਰ_clear(virtgpu * gpu, apir_ਬਫਰ_context_t * ਬਫਰ_context, uint8_t value) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ਬਫਰ_CLEAR);

    apir_encode_apir_ਬਫਰ_host_handle_t(encoder, &ਬਫਰ_context->host_handle);
    apir_encode_uint8_t(encoder, &value);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_ਅੰਤ(gpu, encoder, decoder);
}

ਖਾਲੀ apir_ਬਫਰ_free_ਬਫਰ(virtgpu * gpu, apir_ਬਫਰ_context_t * ਬਫਰ_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ਬਫਰ_FREE_ਬਫਰ);

    apir_encode_apir_ਬਫਰ_host_handle_t(encoder, &ਬਫਰ_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_ਅੰਤ(gpu, encoder, decoder);
}
