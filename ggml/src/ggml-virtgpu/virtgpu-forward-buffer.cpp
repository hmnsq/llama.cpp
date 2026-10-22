#ضمن "virtgpu-forward-impl.h"

فراغ * apir_میانگیر_get_پایه(virtgpu * gpu, apir_میانگیر_context_t * میانگیر_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_میانگیر_GET_پایه);

    apir_encode_apir_میانگیر_host_handle_t(encoder, &میانگیر_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    uintptr_t پایه;
    apir_decode_uintptr_t(decoder, &پایه);

    remote_call_پایان(gpu, encoder, decoder);

    أرجع (فراغ *) پایه;
}

فراغ apir_میانگیر_مجموعه_tensor(virtgpu *               gpu,
                            apir_میانگیر_context_t * میانگیر_context,
                            ggml_tensor *           tensor,
                            ثابت فراغ *            داده,
                            size_t                  جابه_جایی,
                            size_t                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_میانگیر_مجموعه_TENSOR);

    apir_encode_apir_میانگیر_host_handle_t(encoder, &میانگیر_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   موقت_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &موقت_shmem;
    منطقي            using_shared_shmem = خطأ;

    إذا (size <= gpu->داده_shmem.mmap_اندازه) {
        // Lock mutex before using shared data_shmem buffer
        إذا (mtx_قفل(&gpu->داده_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = درست;
        shmem              = &gpu->داده_shmem;

    } وإلا إذا (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    کپی_حافظه(shmem->mmap_ptr, داده, size);
    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);

    apir_encode_اندازه_t(encoder, &جابه_جایی);
    apir_encode_اندازه_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_پایان(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    إذا (using_shared_shmem) {
        mtx_unlock(&gpu->داده_shmem_mutex);
    } وإلا {
        virtgpu_shmem_destroy(gpu, shmem);
    }

    أرجع;
}

فراغ apir_میانگیر_get_tensor(virtgpu *               gpu,
                            apir_میانگیر_context_t * میانگیر_context,
                            ثابت ggml_tensor *     tensor,
                            فراغ *                  داده,
                            size_t                  جابه_جایی,
                            size_t                  size) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_میانگیر_GET_TENSOR);

    apir_encode_apir_میانگیر_host_handle_t(encoder, &میانگیر_context->host_handle);
    apir_encode_ggml_tensor(encoder, tensor);

    virtgpu_shmem   موقت_shmem;  // Local storage for large buffers
    virtgpu_shmem * shmem              = &موقت_shmem;
    منطقي            using_shared_shmem = خطأ;

    إذا (size <= gpu->داده_shmem.mmap_اندازه) {
        // Lock mutex before using shared data_shmem buffer
        إذا (mtx_قفل(&gpu->داده_shmem_mutex) != thrd_success) {
            GGML_ABORT(GGML_VIRTGPU "%s: Failed to lock data_shmem mutex", __func__);
        }
        using_shared_shmem = درست;
        shmem              = &gpu->داده_shmem;

    } وإلا إذا (virtgpu_shmem_create(gpu, size, shmem)) {
        GGML_ABORT(GGML_VIRTGPU "%s: Couldn't allocate the guest-host shared buffer", __func__);
    }

    apir_encode_virtgpu_shmem_res_id(encoder, shmem->res_id);
    apir_encode_اندازه_t(encoder, &جابه_جایی);
    apir_encode_اندازه_t(encoder, &size);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    کپی_حافظه(داده, shmem->mmap_ptr, size);

    remote_call_پایان(gpu, encoder, decoder);

    // Unlock mutex before cleanup
    إذا (using_shared_shmem) {
        mtx_unlock(&gpu->داده_shmem_mutex);
    } وإلا {
        virtgpu_shmem_destroy(gpu, shmem);
    }
}

منطقي apir_میانگیر_cpy_tensor(virtgpu *               gpu,
                            apir_میانگیر_context_t * میانگیر_context,
                            ثابت ggml_tensor *     src,
                            ثابت ggml_tensor *     dst) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_میانگیر_CPY_TENSOR);

    apir_encode_apir_میانگیر_host_handle_t(encoder, &میانگیر_context->host_handle);
    apir_encode_ggml_tensor(encoder, src);
    apir_encode_ggml_tensor(encoder, dst);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    منطقي ret_val;
    apir_decode_bool_t(decoder, &ret_val);

    remote_call_پایان(gpu, encoder, decoder);

    أرجع ret_val;
}

فراغ apir_میانگیر_clear(virtgpu * gpu, apir_میانگیر_context_t * میانگیر_context, uint8_t value) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_میانگیر_CLEAR);

    apir_encode_apir_میانگیر_host_handle_t(encoder, &میانگیر_context->host_handle);
    apir_encode_uint8_t(encoder, &value);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_پایان(gpu, encoder, decoder);
}

فراغ apir_میانگیر_free_میانگیر(virtgpu * gpu, apir_میانگیر_context_t * میانگیر_context) {
    apir_encoder *        encoder;
    apir_decoder *        decoder;
    Apir_Forward_Return_کد ret;

    REMOTE_CALL_PREPARE(gpu, encoder, APIR_COMMAND_TYPE_میانگیر_FREE_میانگیر);

    apir_encode_apir_میانگیر_host_handle_t(encoder, &میانگیر_context->host_handle);

    REMOTE_CALL(gpu, encoder, decoder, ret);

    remote_call_پایان(gpu, encoder, decoder);
}
