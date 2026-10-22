#ضمن "virtgpu-forward-impl.h"

ساكن اوږد اوږد current_وخت_ms() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);  // Use CLOCK_MONOTONIC for elapsed time
    أرجع (اوږد اوږد) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

ggml_حالت apir_backend_ګراف_compute(virtgpu * gpu, ggml_cgraph * cgraph) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کوډ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BACKEND_ګراف_COMPUTE);

    std::vector<uint8_t> cgraph_معلومات;
    size_t               cgraph_اندازه = apir_serialize_ggml_cgraph(cgraph, cgraph_معلومات);

    virtgpu_shmem   لنډمهاله_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &لنډمهاله_shmem;
    منطقي            using_shared_shmem = خطأ;

    إذا (cgraph_اندازه <= gpu->معلومات_shmem.mmap_اندازه) {
        // Lock mutex before using shared data_shmem buffer
        إذا (mtx_قلف(&gpu->معلومات_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = ریښتیا;
        shmem              = &gpu->معلومات_shmem;
    } وإلا إذا (virtgpu_shmem_create(gpu, cgraph_اندازه, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_اندازه_t(encoder, &cgraph_اندازه);

    حرف *       shmem_معلومات    = (حرف *) shmem->mmap_ptr;
    apir_encoder secondary_enc = apir_new_encoder(shmem_معلومات, cgraph_اندازه);

    apir_encode_cgraph_معلومات(&secondary_enc, cgraph_معلومات);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ggml_حالت حالت = GGML_حالت_ABORTED;
    apir_decode_ggml_حالت(decoder, &حالت);

    remote_call_پای(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    إذا (using_shared_shmem) {
        mtx_unlock(&gpu->معلومات_shmem_mutex);
    } وإلا {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    أرجع حالت;
}
