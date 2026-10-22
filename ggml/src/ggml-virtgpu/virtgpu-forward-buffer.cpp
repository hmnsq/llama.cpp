#अन्तर्भाव "virtgpu-forward-impl.h"

शून्य * apir_बफर_get_आधारः(virtgpu * gpu, apir_बफर_context_t * बफर_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_GET_आधारः);

    apir_encode_apir_बफर_host_handle_t(encoder, &बफर_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    uintptr_t आधारः;
    apir_decode_uintptr_t(decoder, &आधारः);

    remote_call_finish(gpu, encoder, decoder);

    प्रत्यागम (शून्य *) आधारः;
}

शून्य apir_बफर_समुच्चयः_tensor(virtgpu *               gpu,
                            apir_बफर_context_t * बफर_context,
                            ggml_tensor *           tensor,
                            ध्रुव शून्य *            दत्त,
                            size_t                  विस्थापनम्,
                            size_t                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_समुच्चयः_TENSOR);

    apir_encode_apir_बफर_host_handle_t(encoder, &बफर_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   अस्थायी_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &अस्थायी_shmem;
    तार्किक            using_shared_shmem = असत्य;

    यदि (size <= gpu->दत्त_shmem.mmap_आकार) {
        // Lock mutex before using shared data_shmem buffer
        यदि (mtx_तालकम्(&gpu->दत्त_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = सत्य;
        shmem              = &gpu->दत्त_shmem;

    } अन्यथा यदि (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    स्मृतिं_प्रतिलिख(shmem->mmap_ptr, दत्त, size);
    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_आकार_t(encoder, &विस्थापनम्);
    apir_encode_आकार_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_finish(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    यदि (using_shared_shmem) {
        mtx_unlock(&gpu->दत्त_shmem_mutex);
    } अन्यथा {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    प्रत्यागम;
}

शून्य apir_बफर_get_tensor(virtgpu *               gpu,
                            apir_बफर_context_t * बफर_context,
                            ध्रुव ggml_tensor *     tensor,
                            शून्य *                  दत्त,
                            size_t                  विस्थापनम्,
                            size_t                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_GET_TENSOR);

    apir_encode_apir_बफर_host_handle_t(encoder, &बफर_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   अस्थायी_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &अस्थायी_shmem;
    तार्किक            using_shared_shmem = असत्य;

    यदि (size <= gpu->दत्त_shmem.mmap_आकार) {
        // Lock mutex before using shared data_shmem buffer
        यदि (mtx_तालकम्(&gpu->दत्त_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = सत्य;
        shmem              = &gpu->दत्त_shmem;

    } अन्यथा यदि (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);
    apir_encode_आकार_t(encoder, &विस्थापनम्);
    apir_encode_आकार_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    स्मृतिं_प्रतिलिख(दत्त, shmem->mmap_ptr, size);

    remote_call_finish(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    यदि (using_shared_shmem) {
        mtx_unlock(&gpu->दत्त_shmem_mutex);
    } अन्यथा {
        virtgpu_shmem_destroy(gpu, shmem);
    }
}

तार्किक apir_बफर_cpy_tensor(virtgpu *               gpu,
                            apir_बफर_context_t * बफर_context,
                            ध्रुव ggml_tensor *     src,
                            ध्रुव ggml_tensor *     dst) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_CPY_TENSOR);

    apir_encode_apir_बफर_host_handle_t(encoder, &बफर_context->host_handle);
    apir_encode_ggml_tensor(encoder, src);
    apir_encode_ggml_tensor(encoder, dst);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    तार्किक ret_val;
    apir_decode_bool_t(decoder, &ret_val);

    remote_call_finish(gpu, encoder, decoder);

    प्रत्यागम ret_val;
}

शून्य apir_बफर_clear(virtgpu * gpu, apir_बफर_context_t * बफर_context, uint8_t value) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_CLEAR);

    apir_encode_apir_बफर_host_handle_t(encoder, &बफर_context->host_handle);
    apir_encode_uint8_t(encoder, &value);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_finish(gpu, encoder, decoder);
}

शून्य apir_बफर_free_बफर(virtgpu * gpu, apir_बफर_context_t * बफर_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_बफर_FREE_बफर);

    apir_encode_apir_बफर_host_handle_t(encoder, &बफर_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_finish(gpu, encoder, decoder);
}
