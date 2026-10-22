#ಅನ್ತರ್ಭಾವ "virtgpu-forward-impl.h"

ಶೂನ್ಯ * apir_ಬಫರ್_get_ತಳ(virtgpu * gpu, apir_ಬಫರ್_context_t * ಬಫರ್_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ಬಫರ್_GET_ತಳ);

    apir_encode_apir_ಬಫರ್_host_handle_t(encoder, &ಬಫರ್_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    uintptr_t ತಳ;
    apir_decode_uintptr_t(decoder, &ತಳ);

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *) ತಳ;
}

ಶೂನ್ಯ apir_ಬಫರ್_ಗಣ_tensor(virtgpu *               gpu,
                            apir_ಬಫರ್_context_t * ಬಫರ್_context,
                            ggml_tensor *           tensor,
                            ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *            ದತ್ತಾಂಶ,
                            size_t                  ಸ್ಥಾನಾಂತರ,
                            size_t                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ಬಫರ್_ಗಣ_TENSOR);

    apir_encode_apir_ಬಫರ್_host_handle_t(encoder, &ಬಫರ್_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   ತಾತ್ಕಾಲಿಕ_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &ತಾತ್ಕಾಲಿಕ_shmem;
    ತಾರ್ಕಿಕ            using_shared_shmem = ಸುಳ್ಳು;

    ಆದರೆ (size <= gpu->ದತ್ತಾಂಶ_shmem.mmap_ಗಾತ್ರ) {
        // Lock mutex before using shared data_shmem buffer
        ಆದರೆ (mtx_ಬೀಗ(&gpu->ದತ್ತಾಂಶ_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = ನಿಜ;
        shmem              = &gpu->ದತ್ತಾಂಶ_shmem;

    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    ಸ್ಮರಣೆ_ನಕಲು(shmem->mmap_ptr, ದತ್ತಾಂಶ, size);
    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_ಗಾತ್ರ_t(encoder, &ಸ್ಥಾನಾಂತರ);
    apir_encode_ಗಾತ್ರ_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    ಆದರೆ (using_shared_shmem) {
        mtx_unlock(&gpu->ದತ್ತಾಂಶ_shmem_mutex);
    } ಇಲ್ಲದಿದ್ದರೆ {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    ಹಿಂತಿರುಗಿ;
}

ಶೂನ್ಯ apir_ಬಫರ್_get_tensor(virtgpu *               gpu,
                            apir_ಬಫರ್_context_t * ಬಫರ್_context,
                            ಸ್ಥಿರಾಂಕ ggml_tensor *     tensor,
                            ಶೂನ್ಯ *                  ದತ್ತಾಂಶ,
                            size_t                  ಸ್ಥಾನಾಂತರ,
                            size_t                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ಬಫರ್_GET_TENSOR);

    apir_encode_apir_ಬಫರ್_host_handle_t(encoder, &ಬಫರ್_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   ತಾತ್ಕಾಲಿಕ_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &ತಾತ್ಕಾಲಿಕ_shmem;
    ತಾರ್ಕಿಕ            using_shared_shmem = ಸುಳ್ಳು;

    ಆದರೆ (size <= gpu->ದತ್ತಾಂಶ_shmem.mmap_ಗಾತ್ರ) {
        // Lock mutex before using shared data_shmem buffer
        ಆದರೆ (mtx_ಬೀಗ(&gpu->ದತ್ತಾಂಶ_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = ನಿಜ;
        shmem              = &gpu->ದತ್ತಾಂಶ_shmem;

    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);
    apir_encode_ಗಾತ್ರ_t(encoder, &ಸ್ಥಾನಾಂತರ);
    apir_encode_ಗಾತ್ರ_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ಸ್ಮರಣೆ_ನಕಲು(ದತ್ತಾಂಶ, shmem->mmap_ptr, size);

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    ಆದರೆ (using_shared_shmem) {
        mtx_unlock(&gpu->ದತ್ತಾಂಶ_shmem_mutex);
    } ಇಲ್ಲದಿದ್ದರೆ {
        virtgpu_shmem_destroy(gpu, shmem);
    }
}

ತಾರ್ಕಿಕ apir_ಬಫರ್_cpy_tensor(virtgpu *               gpu,
                            apir_ಬಫರ್_context_t * ಬಫರ್_context,
                            ಸ್ಥಿರಾಂಕ ggml_tensor *     src,
                            ಸ್ಥಿರಾಂಕ ggml_tensor *     dst) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ಬಫರ್_CPY_TENSOR);

    apir_encode_apir_ಬಫರ್_host_handle_t(encoder, &ಬಫರ್_context->host_handle);
    apir_encode_ggml_tensor(encoder, src);
    apir_encode_ggml_tensor(encoder, dst);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ತಾರ್ಕಿಕ ret_val;
    apir_decode_bool_t(decoder, &ret_val);

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    ಹಿಂತಿರುಗಿ ret_val;
}

ಶೂನ್ಯ apir_ಬಫರ್_clear(virtgpu * gpu, apir_ಬಫರ್_context_t * ಬಫರ್_context, uint8_t value) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ಬಫರ್_CLEAR);

    apir_encode_apir_ಬಫರ್_host_handle_t(encoder, &ಬಫರ್_context->host_handle);
    apir_encode_uint8_t(encoder, &value);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);
}

ಶೂನ್ಯ apir_ಬಫರ್_free_ಬಫರ್(virtgpu * gpu, apir_ಬಫರ್_context_t * ಬಫರ್_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_ಬಫರ್_FREE_ಬಫರ್);

    apir_encode_apir_ಬಫರ್_host_handle_t(encoder, &ಬಫರ್_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);
}
