#ಅನ್ತರ್ಭಾವ "virtgpu-forward-impl.h"

ಸ್ಥಿರ ದೀರ್ಘ ದೀರ್ಘ current_ಸಮಯ_ms() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);  // Use CLOCK_MONOTONIC for elapsed time
    ಹಿಂತಿರುಗಿ (ದೀರ್ಘ ದೀರ್ಘ) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

ggml_ಸ್ಥಿತಿ apir_backend_ನಕ್ಷೆ_compute(virtgpu * gpu, ggml_cgraph * cgraph) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ಸಂಕೇತ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BACKEND_ನಕ್ಷೆ_COMPUTE);

    std::vector<uint8_t> cgraph_ದತ್ತಾಂಶ;
    size_t               cgraph_ಗಾತ್ರ = apir_serialize_ggml_cgraph(cgraph, cgraph_ದತ್ತಾಂಶ);

    virtgpu_shmem   ತಾತ್ಕಾಲಿಕ_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &ತಾತ್ಕಾಲಿಕ_shmem;
    ತಾರ್ಕಿಕ            using_shared_shmem = ಸುಳ್ಳು;

    ಆದರೆ (cgraph_ಗಾತ್ರ <= gpu->ದತ್ತಾಂಶ_shmem.mmap_ಗಾತ್ರ) {
        // Lock mutex before using shared data_shmem buffer
        ಆದರೆ (mtx_ಬೀಗ(&gpu->ದತ್ತಾಂಶ_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = ನಿಜ;
        shmem              = &gpu->ದತ್ತಾಂಶ_shmem;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (virtgpu_shmem_create(gpu, cgraph_ಗಾತ್ರ, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_ಗಾತ್ರ_t(encoder, &cgraph_ಗಾತ್ರ);

    ಅಕ್ಷರ *       shmem_ದತ್ತಾಂಶ    = (ಅಕ್ಷರ *) shmem->mmap_ptr;
    apir_encoder secondary_enc = apir_new_encoder(shmem_ದತ್ತಾಂಶ, cgraph_ಗಾತ್ರ);

    apir_encode_cgraph_ದತ್ತಾಂಶ(&secondary_enc, cgraph_ದತ್ತಾಂಶ);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ggml_ಸ್ಥಿತಿ ಸ್ಥಿತಿ_2 = GGML_ಸ್ಥಿತಿ_ABORTED;
    apir_decode_ggml_ಸ್ಥಿತಿ(decoder, &ಸ್ಥಿತಿ_2);

    remote_call_ಮುಕ್ತಾಯ(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    ಆದರೆ (using_shared_shmem) {
        mtx_unlock(&gpu->ದತ್ತಾಂಶ_shmem_mutex);
    } ಇಲ್ಲದಿದ್ದರೆ {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    ಹಿಂತಿರುಗಿ ಸ್ಥಿತಿ_2;
}
