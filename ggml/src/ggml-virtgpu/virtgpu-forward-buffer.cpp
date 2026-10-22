#شامل "virtgpu-forward-impl.h"

خالی * apir_بفر_get_بنیاد(virtgpu * gpu, apir_بفر_context_t * بفر_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_GET_بنیاد);

    apir_encode_apir_بفر_host_handle_t(encoder, &بفر_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    uintptr_t بنیاد;
    apir_decode_uintptr_t(decoder, &بنیاد);

    remote_call_اختتام(gpu, encoder, decoder);

    واپس (خالی *) بنیاد;
}

خالی apir_بفر_سیٹ_tensor(virtgpu *               gpu,
                            apir_بفر_context_t * بفر_context,
                            ggml_tensor *           tensor,
                            ثابت خالی *            ڈیٹا,
                            size_t                  انحراف,
                            size_t                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_سیٹ_TENSOR);

    apir_encode_apir_بفر_host_handle_t(encoder, &بفر_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   عارضی_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &عارضی_shmem;
    منطقی            using_shared_shmem = جھوٹ;

    اگر (size <= gpu->ڈیٹا_shmem.mmap_سائز) {
        // Lock mutex before using shared data_shmem buffer
        اگر (mtx_تالا(&gpu->ڈیٹا_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = سچ;
        shmem              = &gpu->ڈیٹا_shmem;

    } ورنہ اگر (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    حافظہ_نقل(shmem->mmap_ptr, ڈیٹا, size);
    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_سائز_t(encoder, &انحراف);
    apir_encode_سائز_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_اختتام(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    اگر (using_shared_shmem) {
        mtx_unlock(&gpu->ڈیٹا_shmem_mutex);
    } ورنہ {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    واپس;
}

خالی apir_بفر_get_tensor(virtgpu *               gpu,
                            apir_بفر_context_t * بفر_context,
                            ثابت ggml_tensor *     tensor,
                            خالی *                  ڈیٹا,
                            size_t                  انحراف,
                            size_t                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_GET_TENSOR);

    apir_encode_apir_بفر_host_handle_t(encoder, &بفر_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   عارضی_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &عارضی_shmem;
    منطقی            using_shared_shmem = جھوٹ;

    اگر (size <= gpu->ڈیٹا_shmem.mmap_سائز) {
        // Lock mutex before using shared data_shmem buffer
        اگر (mtx_تالا(&gpu->ڈیٹا_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = سچ;
        shmem              = &gpu->ڈیٹا_shmem;

    } ورنہ اگر (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);
    apir_encode_سائز_t(encoder, &انحراف);
    apir_encode_سائز_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    حافظہ_نقل(ڈیٹا, shmem->mmap_ptr, size);

    remote_call_اختتام(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    اگر (using_shared_shmem) {
        mtx_unlock(&gpu->ڈیٹا_shmem_mutex);
    } ورنہ {
        virtgpu_shmem_destroy(gpu, shmem);
    }
}

منطقی apir_بفر_cpy_tensor(virtgpu *               gpu,
                            apir_بفر_context_t * بفر_context,
                            ثابت ggml_tensor *     src,
                            ثابت ggml_tensor *     dst) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_CPY_TENSOR);

    apir_encode_apir_بفر_host_handle_t(encoder, &بفر_context->host_handle);
    apir_encode_ggml_tensor(encoder, src);
    apir_encode_ggml_tensor(encoder, dst);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    منطقی ret_val;
    apir_decode_bool_t(decoder, &ret_val);

    remote_call_اختتام(gpu, encoder, decoder);

    واپس ret_val;
}

خالی apir_بفر_clear(virtgpu * gpu, apir_بفر_context_t * بفر_context, uint8_t value) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_CLEAR);

    apir_encode_apir_بفر_host_handle_t(encoder, &بفر_context->host_handle);
    apir_encode_uint8_t(encoder, &value);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_اختتام(gpu, encoder, decoder);
}

خالی apir_بفر_free_بفر(virtgpu * gpu, apir_بفر_context_t * بفر_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_بفر_FREE_بفر);

    apir_encode_apir_بفر_host_handle_t(encoder, &بفر_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_اختتام(gpu, encoder, decoder);
}
