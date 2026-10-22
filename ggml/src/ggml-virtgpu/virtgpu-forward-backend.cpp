#ضمن "virtgpu-forward-impl.h"

ساكن دراز دراز current_زمان_ms() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);  // Use CLOCK_MONOTONIC for elapsed time
    أرجع (دراز دراز) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

ggml_وضعیت apir_backend_گراف_compute(virtgpu * gpu, ggml_cgraph * cgraph) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BACKEND_گراف_COMPUTE);

    std::vector<uint8_t> cgraph_داده;
    size_t               cgraph_اندازه = apir_serialize_ggml_cgraph(cgraph, cgraph_داده);

    virtgpu_shmem   موقت_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &موقت_shmem;
    منطقي            using_shared_shmem = خطأ;

    إذا (cgraph_اندازه <= gpu->داده_shmem.mmap_اندازه) {
        // Lock mutex before using shared data_shmem buffer
        إذا (mtx_قفل(&gpu->داده_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = درست;
        shmem              = &gpu->داده_shmem;
    } وإلا إذا (virtgpu_shmem_create(gpu, cgraph_اندازه, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_اندازه_t(encoder, &cgraph_اندازه);

    حرف *       shmem_داده    = (حرف *) shmem->mmap_ptr;
    apir_encoder secondary_enc = apir_new_encoder(shmem_داده, cgraph_اندازه);

    apir_encode_cgraph_داده(&secondary_enc, cgraph_داده);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ggml_وضعیت وضعیت = GGML_وضعیت_ABORTED;
    apir_decode_ggml_وضعیت(decoder, &وضعیت);

    remote_call_پایان(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    إذا (using_shared_shmem) {
        mtx_unlock(&gpu->داده_shmem_mutex);
    } وإلا {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    أرجع وضعیت;
}
