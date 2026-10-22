#ਅਨ੍ਤਰ੍ਭਾਵ "virtgpu-forward-impl.h"

ਸਥਿਰ ਲੰਮਾ ਲੰਮਾ current_ਸਮਾਂ_ms() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);  // Use CLOCK_MONOTONIC for elapsed time
    ਵਾਪਸ (ਲੰਮਾ ਲੰਮਾ) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

ggml_ਸਥਿਤੀ apir_backend_ਗ੍ਰਾਫ_compute(virtgpu * gpu, ggml_cgraph * cgraph) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ਕੋਡ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BACKEND_ਗ੍ਰਾਫ_COMPUTE);

    std::vector<uint8_t> cgraph_ਅੰਕੜੇ;
    size_t               cgraph_ਮਾਪ = apir_serialize_ggml_cgraph(cgraph, cgraph_ਅੰਕੜੇ);

    virtgpu_shmem   ਆਰਜ਼ੀ_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &ਆਰਜ਼ੀ_shmem;
    ਤਾਰਕਿਕ            using_shared_shmem = ਝੂਠ;

    ਜੇ (cgraph_ਮਾਪ <= gpu->ਅੰਕੜੇ_shmem.mmap_ਮਾਪ) {
        // Lock mutex before using shared data_shmem buffer
        ਜੇ (mtx_ਤਾਲਾ(&gpu->ਅੰਕੜੇ_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = ਸੱਚ;
        shmem              = &gpu->ਅੰਕੜੇ_shmem;
    } ਨਹੀਂਤਾਂ ਜੇ (virtgpu_shmem_create(gpu, cgraph_ਮਾਪ, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_ਮਾਪ_t(encoder, &cgraph_ਮਾਪ);

    ਅੱਖਰ *       shmem_ਅੰਕੜੇ    = (ਅੱਖਰ *) shmem->mmap_ptr;
    apir_encoder secondary_enc = apir_new_encoder(shmem_ਅੰਕੜੇ, cgraph_ਮਾਪ);

    apir_encode_cgraph_ਅੰਕੜੇ(&secondary_enc, cgraph_ਅੰਕੜੇ);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ggml_ਸਥਿਤੀ ਸਥਿਤੀ = GGML_ਸਥਿਤੀ_ABORTED;
    apir_decode_ggml_ਸਥਿਤੀ(decoder, &ਸਥਿਤੀ);

    remote_call_ਅੰਤ(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    ਜੇ (using_shared_shmem) {
        mtx_unlock(&gpu->ਅੰਕੜੇ_shmem_mutex);
    } ਨਹੀਂਤਾਂ {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    ਵਾਪਸ ਸਥਿਤੀ;
}
