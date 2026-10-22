#ضمن "virtgpu-forward-impl.h"

فراغ * apir_بفر_get_بنياد(virtgpu * gpu, apir_بفر_context_t * بفر_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ڪوڊ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_GET_بنياد);

    apir_encode_apir_بفر_host_handle_t(encoder, &بفر_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    uintptr_t بنياد;
    apir_decode_uintptr_t(decoder, &بنياد);

    remote_call_پڄاڻي(gpu, encoder, decoder);

    أرجع (فراغ *) بنياد;
}

فراغ apir_بفر_سيٽ_tensor(virtgpu *               gpu,
                            apir_بفر_context_t * بفر_context,
                            ggml_tensor *           tensor,
                            ثابت فراغ *            معلومات,
                            size_t                  وٿي,
                            size_t                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ڪوڊ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_سيٽ_TENSOR);

    apir_encode_apir_بفر_host_handle_t(encoder, &بفر_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   عارضي_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &عارضي_shmem;
    منطقي            using_shared_shmem = خطأ;

    إذا (size <= gpu->معلومات_shmem.mmap_سائيز) {
        // Lock mutex before using shared data_shmem buffer
        إذا (mtx_تالو(&gpu->معلومات_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = سچ;
        shmem              = &gpu->معلومات_shmem;

    } وإلا إذا (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    يادگيري_نقل(shmem->mmap_ptr, معلومات, size);
    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_سائيز_t(encoder, &وٿي);
    apir_encode_سائيز_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_پڄاڻي(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    إذا (using_shared_shmem) {
        mtx_unlock(&gpu->معلومات_shmem_mutex);
    } وإلا {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    أرجع;
}

فراغ apir_بفر_get_tensor(virtgpu *               gpu,
                            apir_بفر_context_t * بفر_context,
                            ثابت ggml_tensor *     tensor,
                            فراغ *                  معلومات,
                            size_t                  وٿي,
                            size_t                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ڪوڊ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_GET_TENSOR);

    apir_encode_apir_بفر_host_handle_t(encoder, &بفر_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   عارضي_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &عارضي_shmem;
    منطقي            using_shared_shmem = خطأ;

    إذا (size <= gpu->معلومات_shmem.mmap_سائيز) {
        // Lock mutex before using shared data_shmem buffer
        إذا (mtx_تالو(&gpu->معلومات_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = سچ;
        shmem              = &gpu->معلومات_shmem;

    } وإلا إذا (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);
    apir_encode_سائيز_t(encoder, &وٿي);
    apir_encode_سائيز_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    يادگيري_نقل(معلومات, shmem->mmap_ptr, size);

    remote_call_پڄاڻي(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    إذا (using_shared_shmem) {
        mtx_unlock(&gpu->معلومات_shmem_mutex);
    } وإلا {
        virtgpu_shmem_destroy(gpu, shmem);
    }
}

منطقي apir_بفر_cpy_tensor(virtgpu *               gpu,
                            apir_بفر_context_t * بفر_context,
                            ثابت ggml_tensor *     src,
                            ثابت ggml_tensor *     dst) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ڪوڊ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_CPY_TENSOR);

    apir_encode_apir_بفر_host_handle_t(encoder, &بفر_context->host_handle);
    apir_encode_ggml_tensor(encoder, src);
    apir_encode_ggml_tensor(encoder, dst);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    منطقي ret_val;
    apir_decode_bool_t(decoder, &ret_val);

    remote_call_پڄاڻي(gpu, encoder, decoder);

    أرجع ret_val;
}

فراغ apir_بفر_clear(virtgpu * gpu, apir_بفر_context_t * بفر_context, uint8_t value) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ڪوڊ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_CLEAR);

    apir_encode_apir_بفر_host_handle_t(encoder, &بفر_context->host_handle);
    apir_encode_uint8_t(encoder, &value);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_پڄاڻي(gpu, encoder, decoder);
}

فراغ apir_بفر_free_بفر(virtgpu * gpu, apir_بفر_context_t * بفر_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ڪوڊ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_FREE_بفر);

    apir_encode_apir_بفر_host_handle_t(encoder, &بفر_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_پڄاڻي(gpu, encoder, decoder);
}
