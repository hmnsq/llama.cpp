#ضمن "virtgpu-forward-impl.h"

ساكن لما لما current_ویلا_ms() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);  // Use CLOCK_MONOTONIC for elapsed time
    أرجع (لما لما) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

ggml_حالت apir_backend_گراف_compute(virtgpu * gpu, ggml_cgraph * cgraph) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کوڈ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BACKEND_گراف_COMPUTE);

    std::vector<uint8_t> cgraph_انکڑے;
    size_t               cgraph_ناپ = apir_serialize_ggml_cgraph(cgraph, cgraph_انکڑے);

    virtgpu_shmem   عارضی_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &عارضی_shmem;
    منطقي            using_shared_shmem = خطأ;

    إذا (cgraph_ناپ <= gpu->انکڑے_shmem.mmap_ناپ) {
        // Lock mutex before using shared data_shmem buffer
        إذا (mtx_جندرا(&gpu->انکڑے_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = سچ;
        shmem              = &gpu->انکڑے_shmem;
    } وإلا إذا (virtgpu_shmem_create(gpu, cgraph_ناپ, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_ناپ_t(encoder, &cgraph_ناپ);

    حرف *       shmem_انکڑے    = (حرف *) shmem->mmap_ptr;
    apir_encoder secondary_enc = apir_new_encoder(shmem_انکڑے, cgraph_ناپ);

    apir_encode_cgraph_انکڑے(&secondary_enc, cgraph_انکڑے);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ggml_حالت حالت = GGML_حالت_ABORTED;
    apir_decode_ggml_حالت(decoder, &حالت);

    remote_call_انت(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    إذا (using_shared_shmem) {
        mtx_unlock(&gpu->انکڑے_shmem_mutex);
    } وإلا {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    أرجع حالت;
}
