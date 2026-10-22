#অন্তর্ভাব "virtgpu-forward-impl.h"

স্থির long long current_time_ms() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);  // Use CLOCK_MONOTONIC for elapsed time
    ফেরত (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

ggml_status apir_backend_graph_compute(virtgpu * gpu, ggml_cgraph * cgraph) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    ApirForwardReturnCode ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_BACKEND_GRAPH_COMPUTE);

    std::vector<uint8_t> cgraph_তথ্য;
    size_t               cgraph_আকার = apir_serialize_ggml_cgraph(cgraph, cgraph_তথ্য);

    virtgpu_shmem   অস্থায়ী_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &অস্থায়ী_shmem;
    যৌক্তিক            using_shared_shmem = মিথ্যা;

    যদি (cgraph_আকার <= gpu->তথ্য_shmem.mmap_আকার) {
        // Lock mutex before using shared data_shmem buffer
        যদি (mtx_lock(&gpu->তথ্য_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = সত্য;
        shmem              = &gpu->তথ্য_shmem;
    } অন্যথা যদি (virtgpu_shmem_create(gpu, cgraph_আকার, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_আকার_t(encoder, &cgraph_আকার);

    অক্ষর *       shmem_তথ্য    = (অক্ষর *) shmem->mmap_ptr;
    apir_encoder secondary_enc = apir_new_encoder(shmem_তথ্য, cgraph_আকার);

    apir_encode_cgraph_তথ্য(&secondary_enc, cgraph_তথ্য);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    ggml_status status = GGML_STATUS_ABORTED;
    apir_decode_ggml_status(decoder, &status);

    remote_call_finish(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    যদি (using_shared_shmem) {
        mtx_unlock(&gpu->তথ্য_shmem_mutex);
    } অন্যথা {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    ফেরত status;
}
