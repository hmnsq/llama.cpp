#ضمن "virtgpu-forward-impl.h"

فراغ * apir_بفر_get_ادھار(virtgpu * gpu, apir_بفر_context_t * بفر_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کوڈ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_GET_ادھار);

    apir_encode_apir_بفر_host_handle_t(encoder, &بفر_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    uintptr_t ادھار;
    apir_decode_uintptr_t(decoder, &ادھار);

    remote_call_انت(gpu, encoder, decoder);

    أرجع (فراغ *) ادھار;
}

فراغ apir_بفر_سیٹ_tensor(virtgpu *               gpu,
                            apir_بفر_context_t * بفر_context,
                            ggml_tensor *           tensor,
                            ثابت فراغ *            انکڑے,
                            size_t                  وتھ,
                            size_t                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کوڈ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_سیٹ_TENSOR);

    apir_encode_apir_بفر_host_handle_t(encoder, &بفر_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   عارضی_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &عارضی_shmem;
    منطقي            using_shared_shmem = خطأ;

    إذا (size <= gpu->انکڑے_shmem.mmap_ناپ) {
        // Lock mutex before using shared data_shmem buffer
        إذا (mtx_جندرا(&gpu->انکڑے_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = سچ;
        shmem              = &gpu->انکڑے_shmem;

    } وإلا إذا (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    حافظہ_نقل(shmem->mmap_ptr, انکڑے, size);
    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_ناپ_t(encoder, &وتھ);
    apir_encode_ناپ_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_انت(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    إذا (using_shared_shmem) {
        mtx_unlock(&gpu->انکڑے_shmem_mutex);
    } وإلا {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    أرجع;
}

فراغ apir_بفر_get_tensor(virtgpu *               gpu,
                            apir_بفر_context_t * بفر_context,
                            ثابت ggml_tensor *     tensor,
                            فراغ *                  انکڑے,
                            size_t                  وتھ,
                            size_t                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کوڈ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_GET_TENSOR);

    apir_encode_apir_بفر_host_handle_t(encoder, &بفر_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   عارضی_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &عارضی_shmem;
    منطقي            using_shared_shmem = خطأ;

    إذا (size <= gpu->انکڑے_shmem.mmap_ناپ) {
        // Lock mutex before using shared data_shmem buffer
        إذا (mtx_جندرا(&gpu->انکڑے_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = سچ;
        shmem              = &gpu->انکڑے_shmem;

    } وإلا إذا (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);
    apir_encode_ناپ_t(encoder, &وتھ);
    apir_encode_ناپ_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    حافظہ_نقل(انکڑے, shmem->mmap_ptr, size);

    remote_call_انت(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    إذا (using_shared_shmem) {
        mtx_unlock(&gpu->انکڑے_shmem_mutex);
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
    Apir_Forward_Return_کوڈ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_CPY_TENSOR);

    apir_encode_apir_بفر_host_handle_t(encoder, &بفر_context->host_handle);
    apir_encode_ggml_tensor(encoder, src);
    apir_encode_ggml_tensor(encoder, dst);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    منطقي ret_val;
    apir_decode_bool_t(decoder, &ret_val);

    remote_call_انت(gpu, encoder, decoder);

    أرجع ret_val;
}

فراغ apir_بفر_clear(virtgpu * gpu, apir_بفر_context_t * بفر_context, uint8_t value) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کوڈ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_CLEAR);

    apir_encode_apir_بفر_host_handle_t(encoder, &بفر_context->host_handle);
    apir_encode_uint8_t(encoder, &value);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_انت(gpu, encoder, decoder);
}

فراغ apir_بفر_free_بفر(virtgpu * gpu, apir_بفر_context_t * بفر_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کوڈ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_FREE_بفر);

    apir_encode_apir_بفر_host_handle_t(encoder, &بفر_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_انت(gpu, encoder, decoder);
}
