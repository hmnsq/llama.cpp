#समावेश "virtgpu-forward-impl.h"

अटल दीर्घ दीर्घ current_time_ms() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);  // Use CLOCK_MONOTONIC for elapsed time
    वापस (दीर्घ दीर्घ) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

ggml_status apir_backend_graph_compute(virtgpu * gpu, ggml_cgraph * cgraph) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE);

    std::vector<uint8_t> cgraph_तथ्य;
    माप_प्रकार               cgraph_आकार = apir_serialize_ggml_cgraph(cgraph, cgraph_तथ्य);

    virtgpu_shmem   अस्थायी_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &अस्थायी_shmem;
    bool            using_shared_shmem = false;

    अगर (cgraph_आकार <= gpu->तथ्य_shmem.mmap_आकार) {
        // Lock mutex before using shared data_shmem buffer
        अगर (mtx_lock(&gpu->तथ्य_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = true;
        shmem              = &gpu->तथ्य_shmem;
    } अन्यथा अगर (virtgpu_shmem_create(gpu, cgraph_आकार, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_आकार_t(encoder, &cgraph_आकार);

    अक्षर *       shmem_तथ्य    = (अक्षर *) shmem->mmap_ptr;
    apir_encoder secondary_enc = apir_new_encoder(shmem_तथ्य, cgraph_आकार);

    apir_encode_cgraph_तथ्य(&secondary_enc, cgraph_तथ्य);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ggml_status status = GGML_STATUS_ABORTED;
    apir_decode_ggml_status(decoder, &status);

    remote_call_finish(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    अगर (using_shared_shmem) {
        mtx_unlock(&gpu->तथ्य_shmem_mutex);
    } अन्यथा {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    वापस status;
}
