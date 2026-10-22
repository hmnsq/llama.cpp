#అన్తర్భావ "virtgpu-forward-impl.h"

స్థిర దీర్ఘ దీర్ఘ current_సమయం_ms() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);  // Use CLOCK_MONOTONIC for elapsed time
    తిరిగి (దీర్ఘ దీర్ఘ) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

ggml_స్థితి apir_backend_లేఖాచిత్రం_compute(virtgpu * gpu, ggml_cgraph * cgraph) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BACKEND_లేఖాచిత్రం_COMPUTE);

    std::vector<uint8_t> cgraph_దత్తాంశం;
    size_t               cgraph_కొలత = apir_serialize_ggml_cgraph(cgraph, cgraph_దత్తాంశం);

    virtgpu_shmem   తాత్కాలిక_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &తాత్కాలిక_shmem;
    తార్కిక            using_shared_shmem = అబద్ధం;

    ఒకవేళ (cgraph_కొలత <= gpu->దత్తాంశం_shmem.mmap_కొలత) {
        // Lock mutex before using shared data_shmem buffer
        ఒకవేళ (mtx_తాళం(&gpu->దత్తాంశం_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = నిజం;
        shmem              = &gpu->దత్తాంశం_shmem;
    } లేకపోతే ఒకవేళ (virtgpu_shmem_create(gpu, cgraph_కొలత, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_కొలత_t(encoder, &cgraph_కొలత);

    అక్షరం *       shmem_దత్తాంశం    = (అక్షరం *) shmem->mmap_ptr;
    apir_encoder secondary_enc = apir_new_encoder(shmem_దత్తాంశం, cgraph_కొలత);

    apir_encode_cgraph_దత్తాంశం(&secondary_enc, cgraph_దత్తాంశం);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ggml_స్థితి స్థితి_2 = GGML_స్థితి_ABORTED;
    apir_decode_ggml_స్థితి(decoder, &స్థితి_2);

    remote_call_ముగింపు(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    ఒకవేళ (using_shared_shmem) {
        mtx_unlock(&gpu->దత్తాంశం_shmem_mutex);
    } లేకపోతే {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    తిరిగి స్థితి_2;
}
