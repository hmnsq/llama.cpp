#अन्तर्भाव "virtgpu-forward-impl.h"

स्थिर दीर्घ दीर्घ current_time_ms() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);  // Use CLOCK_MONOTONIC for elapsed time
    प्रत्यागम (दीर्घ दीर्घ) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

ggml_स्थितिः apir_backend_graph_compute(virtgpu * gpu, ggml_cgraph * cgraph) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_संहिता ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE);

    std::vector<uint8_t> cgraph_दत्त;
    size_t               cgraph_आकार = apir_serialize_ggml_cgraph(cgraph, cgraph_दत्त);

    virtgpu_shmem   अस्थायी_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &अस्थायी_shmem;
    तार्किक            using_shared_shmem = असत्य;

    यदि (cgraph_आकार <= gpu->दत्त_shmem.mmap_आकार) {
        // Lock mutex before using shared data_shmem buffer
        यदि (mtx_तालकम्(&gpu->दत्त_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = सत्य;
        shmem              = &gpu->दत्त_shmem;
    } अन्यथा यदि (virtgpu_shmem_create(gpu, cgraph_आकार, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_आकार_t(encoder, &cgraph_आकार);

    वर्ण *       shmem_दत्त    = (वर्ण *) shmem->mmap_ptr;
    apir_encoder secondary_enc = apir_new_encoder(shmem_दत्त, cgraph_आकार);

    apir_encode_cgraph_दत्त(&secondary_enc, cgraph_दत्त);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ggml_स्थितिः स्थितिः = GGML_स्थितिः_ABORTED;
    apir_decode_ggml_स्थितिः(decoder, &स्थितिः);

    remote_call_finish(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    यदि (using_shared_shmem) {
        mtx_unlock(&gpu->दत्त_shmem_mutex);
    } अन्यथा {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    प्रत्यागम स्थितिः;
}
