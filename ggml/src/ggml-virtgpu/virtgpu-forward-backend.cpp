#ضمن "virtgpu-forward-impl.h"

ساكن ڊگهو ڊگهو current_وقت_ms() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);  // Use CLOCK_MONOTONIC for elapsed time
    أرجع (ڊگهو ڊگهو) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

ggml_حالت apir_backend_گراف_compute(virtgpu * gpu, ggml_cgraph * cgraph) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_ڪوڊ ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BACKEND_گراف_COMPUTE);

    std::vector<uint8_t> cgraph_معلومات;
    size_t               cgraph_سائيز = apir_serialize_ggml_cgraph(cgraph, cgraph_معلومات);

    virtgpu_shmem   عارضي_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &عارضي_shmem;
    منطقي            using_shared_shmem = خطأ;

    إذا (cgraph_سائيز <= gpu->معلومات_shmem.mmap_سائيز) {
        // Lock mutex before using shared data_shmem buffer
        إذا (mtx_تالو(&gpu->معلومات_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = سچ;
        shmem              = &gpu->معلومات_shmem;
    } وإلا إذا (virtgpu_shmem_create(gpu, cgraph_سائيز, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_سائيز_t(encoder, &cgraph_سائيز);

    حرف *       shmem_معلومات    = (حرف *) shmem->mmap_ptr;
    apir_encoder secondary_enc = apir_new_encoder(shmem_معلومات, cgraph_سائيز);

    apir_encode_cgraph_معلومات(&secondary_enc, cgraph_معلومات);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ggml_حالت حالت = GGML_حالت_ABORTED;
    apir_decode_ggml_حالت(decoder, &حالت);

    remote_call_پڄاڻي(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    إذا (using_shared_shmem) {
        mtx_unlock(&gpu->معلومات_shmem_mutex);
    } وإلا {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    أرجع حالت;
}
