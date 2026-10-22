#ਅਨ੍ਤਰ੍ਭਾਵ "backend-dispatched.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "backend-virgl-apir.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "shared/apir_backend.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>

ਸਥਿਰ uint32_t validate_ਗ੍ਰਾਫ_operation(size_t cgraph_ਮਾਪ, uint32_t shmem_res_id, ਅਚਲ ਅੱਖਰ * operation) {
    ਜੇ (cgraph_ਮਾਪ == 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Zero-size computation graph\n", operation);
        ਵਾਪਸ 1;
    }

    // place-holder: validate that the size of shmem_res_id is <= cgraph_size
    // need to add another method in the Virgl->APIR callback interface
    GGML_ਅਣਵਰਤਿਆ(shmem_res_id);

    ਵਾਪਸ 0;  // Valid
}

uint32_t backend_backend_ਗ੍ਰਾਫ_compute(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_ਅਣਵਰਤਿਆ(ctx);

    ਸਥਿਰ ਤਾਰਕਿਕ async_backend_initialized = ਝੂਠ;
    ਸਥਿਰ ਤਾਰਕਿਕ async_backend;

    ਜੇ (!async_backend_initialized) {
        ggml_backend_dev_props props;

        dev->iface.get_props(dev, &props);
        async_backend             = props.caps.async;
        async_backend_initialized = ਸੱਚ;
    }

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    ਅਚਲ ਖਾਲੀ * shmem_ਅੰਕੜੇ = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    ਜੇ (!shmem_ਅੰਕੜੇ) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_ਸੈੱਟ_fatal(dec);
        ਵਾਪਸ 1;
    }
    size_t cgraph_ਮਾਪ;
    apir_decode_ਮਾਪ_t(dec, &cgraph_ਮਾਪ);

    ਜੇ (validate_ਗ੍ਰਾਫ_operation(cgraph_ਮਾਪ, shmem_res_id, __func__) != 0) {
        apir_decoder_ਸੈੱਟ_fatal(dec);
        ਵਾਪਸ 1;
    }

    apir_decoder secondary_dec = apir_new_decoder((ਅਚਲ ਅੱਖਰ *) shmem_ਅੰਕੜੇ, cgraph_ਮਾਪ);

    ggml_cgraph * cgraph = apir_decode_ggml_cgraph(&secondary_dec, cgraph_ਮਾਪ);

    ਜੇ (!cgraph || apir_decoder_get_fatal(&secondary_dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to deserialize computation graph\n", __func__);
        ਵਾਪਸ 1;
    }

    ਜੇ (cgraph->n_nodes < 0 || cgraph->n_leafs < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid negative node/leaf count: nodes=%d leafs=%d\n", __func__,
                       cgraph->n_nodes, cgraph->n_leafs);
        ਵਾਪਸ 1;
    }

    ggml_ਸਥਿਤੀ ਸਥਿਤੀ;
#ਜੇ APIR_BACKEND_CHECK_SUPPORTS_OP == 1
    ਲਈ (ਪੂਰਨ idx = 0; idx < cgraph->n_nodes; idx++) {
        ggml_tensor * op = ggml_ਗ੍ਰਾਫ_ਗੰਢ(cgraph, idx);
        ਜੇ (dev->iface.supports_op(dev, op)) {
            ਜਾਰੀ;
        }
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Graph node %d (%s) not supported by the backend\n", __func__, idx,
                       ggml_op_desc(op));

        ਸਥਿਤੀ = GGML_ਸਥਿਤੀ_ABORTED;
        apir_encode_ggml_ਸਥਿਤੀ(enc, &ਸਥਿਤੀ);

        ਵਾਪਸ 0;
    }
#ਜੇ_ਅੰਤ

    // Check if backend is properly initialized
    ਜੇ (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Backend not initialized (bck is null)\n", __func__);

        ਵਾਪਸ 1;
    }

    ਸਥਿਤੀ = bck->iface.ਗ੍ਰਾਫ_compute(bck, cgraph);

    ਜੇ (async_backend && bck->iface.synchronize) {
        bck->iface.synchronize(bck);
    }

    apir_encode_ggml_ਸਥਿਤੀ(enc, &ਸਥਿਤੀ);

    ਵਾਪਸ 0;
}
