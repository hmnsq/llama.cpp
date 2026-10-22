#شامل "virtgpu-forward-impl.h"

ساکن طویل طویل current_وقت_ms() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);  // Use CLOCK_MONOTONIC for elapsed time
    واپس (طویل طویل) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

ggml_حالت apir_backend_گراف_compute(virtgpu * gpu, ggml_cgraph * cgraph) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_رمز ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BACKEND_گراف_COMPUTE);

    std::vector<uint8_t> cgraph_ڈیٹا;
    size_t               cgraph_سائز = apir_serialize_ggml_cgraph(cgraph, cgraph_ڈیٹا);

    virtgpu_shmem   عارضی_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &عارضی_shmem;
    منطقی            using_shared_shmem = جھوٹ;

    اگر (cgraph_سائز <= gpu->ڈیٹا_shmem.mmap_سائز) {
        // Lock mutex before using shared data_shmem buffer
        اگر (mtx_تالا(&gpu->ڈیٹا_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = سچ;
        shmem              = &gpu->ڈیٹا_shmem;
    } ورنہ اگر (virtgpu_shmem_create(gpu, cgraph_سائز, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_سائز_t(encoder, &cgraph_سائز);

    حرف *       shmem_ڈیٹا    = (حرف *) shmem->mmap_ptr;
    apir_encoder secondary_enc = apir_new_encoder(shmem_ڈیٹا, cgraph_سائز);

    apir_encode_cgraph_ڈیٹا(&secondary_enc, cgraph_ڈیٹا);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ggml_حالت حالت_2 = GGML_حالت_ABORTED;
    apir_decode_ggml_حالت(decoder, &حالت_2);

    remote_call_اختتام(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    اگر (using_shared_shmem) {
        mtx_unlock(&gpu->ڈیٹا_shmem_mutex);
    } ورنہ {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    واپس حالت_2;
}
