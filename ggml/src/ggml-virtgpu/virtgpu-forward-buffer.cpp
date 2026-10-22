#అన్తర్భావ "virtgpu-forward-impl.h"

శూన్య * apir_బఫరు_get_పునాది(virtgpu * gpu, apir_బఫరు_context_t * బఫరు_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_బఫరు_GET_పునాది);

    apir_encode_apir_బఫరు_host_handle_t(encoder, &బఫరు_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    uintptr_t పునాది;
    apir_decode_uintptr_t(decoder, &పునాది);

    remote_call_ముగింపు(gpu, encoder, decoder);

    తిరిగి (శూన్య *) పునాది;
}

శూన్య apir_బఫరు_సమితి_tensor(virtgpu *               gpu,
                            apir_బఫరు_context_t * బఫరు_context,
                            ggml_tensor *           tensor,
                            స్థిరరాశి శూన్య *            దత్తాంశం,
                            size_t                  స్థానభ్రంశం,
                            size_t                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_బఫరు_సమితి_TENSOR);

    apir_encode_apir_బఫరు_host_handle_t(encoder, &బఫరు_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   తాత్కాలిక_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &తాత్కాలిక_shmem;
    తార్కిక            using_shared_shmem = అబద్ధం;

    ఒకవేళ (size <= gpu->దత్తాంశం_shmem.mmap_కొలత) {
        // Lock mutex before using shared data_shmem buffer
        ఒకవేళ (mtx_తాళం(&gpu->దత్తాంశం_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = నిజం;
        shmem              = &gpu->దత్తాంశం_shmem;

    } లేకపోతే ఒకవేళ (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    జ్ఞాపకం_నకలు(shmem->mmap_ptr, దత్తాంశం, size);
    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_కొలత_t(encoder, &స్థానభ్రంశం);
    apir_encode_కొలత_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_ముగింపు(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    ఒకవేళ (using_shared_shmem) {
        mtx_unlock(&gpu->దత్తాంశం_shmem_mutex);
    } లేకపోతే {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    తిరిగి;
}

శూన్య apir_బఫరు_get_tensor(virtgpu *               gpu,
                            apir_బఫరు_context_t * బఫరు_context,
                            స్థిరరాశి ggml_tensor *     tensor,
                            శూన్య *                  దత్తాంశం,
                            size_t                  స్థానభ్రంశం,
                            size_t                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_బఫరు_GET_TENSOR);

    apir_encode_apir_బఫరు_host_handle_t(encoder, &బఫరు_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   తాత్కాలిక_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &తాత్కాలిక_shmem;
    తార్కిక            using_shared_shmem = అబద్ధం;

    ఒకవేళ (size <= gpu->దత్తాంశం_shmem.mmap_కొలత) {
        // Lock mutex before using shared data_shmem buffer
        ఒకవేళ (mtx_తాళం(&gpu->దత్తాంశం_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = నిజం;
        shmem              = &gpu->దత్తాంశం_shmem;

    } లేకపోతే ఒకవేళ (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);
    apir_encode_కొలత_t(encoder, &స్థానభ్రంశం);
    apir_encode_కొలత_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    జ్ఞాపకం_నకలు(దత్తాంశం, shmem->mmap_ptr, size);

    remote_call_ముగింపు(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    ఒకవేళ (using_shared_shmem) {
        mtx_unlock(&gpu->దత్తాంశం_shmem_mutex);
    } లేకపోతే {
        virtgpu_shmem_destroy(gpu, shmem);
    }
}

తార్కిక apir_బఫరు_cpy_tensor(virtgpu *               gpu,
                            apir_బఫరు_context_t * బఫరు_context,
                            స్థిరరాశి ggml_tensor *     src,
                            స్థిరరాశి ggml_tensor *     dst) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_బఫరు_CPY_TENSOR);

    apir_encode_apir_బఫరు_host_handle_t(encoder, &బఫరు_context->host_handle);
    apir_encode_ggml_tensor(encoder, src);
    apir_encode_ggml_tensor(encoder, dst);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    తార్కిక ret_val;
    apir_decode_bool_t(decoder, &ret_val);

    remote_call_ముగింపు(gpu, encoder, decoder);

    తిరిగి ret_val;
}

శూన్య apir_బఫరు_clear(virtgpu * gpu, apir_బఫరు_context_t * బఫరు_context, uint8_t value) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_బఫరు_CLEAR);

    apir_encode_apir_బఫరు_host_handle_t(encoder, &బఫరు_context->host_handle);
    apir_encode_uint8_t(encoder, &value);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_ముగింపు(gpu, encoder, decoder);
}

శూన్య apir_బఫరు_free_బఫరు(virtgpu * gpu, apir_బఫరు_context_t * బఫరు_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_సంకేతం ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_బఫరు_FREE_బఫరు);

    apir_encode_apir_బఫరు_host_handle_t(encoder, &బఫరు_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_ముగింపు(gpu, encoder, decoder);
}
