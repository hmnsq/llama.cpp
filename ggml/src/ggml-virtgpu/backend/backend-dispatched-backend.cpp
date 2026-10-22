#அந்தர்பாவ "backend-dispatched.h"
#அந்தர்பாவ "backend-virgl-apir.h"
#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-backend.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "shared/apir_backend.h"

#அந்தர்பாவ <cstdint>

நிலையான uint32_t validate_graph_operation(size_t cgraph_size, uint32_t shmem_res_id, மாறிலி எழுத்து * operation) {
    எனில் (cgraph_size == 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Zero-size computation graph\n", operation);
        திரும்பு 1;
    }

    // place-holder: validate that the size of shmem_res_id is <= cgraph_size
    // need to add another method in the Virgl->APIR callback interface
    GGML_UNUSED(shmem_res_id);

    திரும்பு 0;  // Valid
}

uint32_t backend_backend_graph_compute(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);

    நிலையான தருக்க async_backend_initialized = பொய்;
    நிலையான தருக்க async_backend;

    எனில் (!async_backend_initialized) {
        ggml_backend_dev_props props;

        dev->iface.get_props(dev, &props);
        async_backend             = props.caps.async;
        async_backend_initialized = உண்மை;
    }

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    மாறிலி வெறுமை * shmem_data = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    எனில் (!shmem_data) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_set_fatal(dec);
        திரும்பு 1;
    }
    size_t cgraph_size;
    apir_decode_size_t(dec, &cgraph_size);

    எனில் (validate_graph_operation(cgraph_size, shmem_res_id, __func__) != 0) {
        apir_decoder_set_fatal(dec);
        திரும்பு 1;
    }

    apir_decoder secondary_dec = apir_new_decoder((மாறிலி எழுத்து *) shmem_data, cgraph_size);

    ggml_cgraph * cgraph = apir_decode_ggml_cgraph(&secondary_dec, cgraph_size);

    எனில் (!cgraph || apir_decoder_get_fatal(&secondary_dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to deserialize computation graph\n", __func__);
        திரும்பு 1;
    }

    எனில் (cgraph->n_nodes < 0 || cgraph->n_leafs < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid negative node/leaf count: nodes=%d leafs=%d\n", __func__,
                       cgraph->n_nodes, cgraph->n_leafs);
        திரும்பு 1;
    }

    ggml_status status;
#எனில் APIR_BACKEND_CHECK_SUPPORTS_OP == 1
    க்கு (முழு idx = 0; idx < cgraph->n_nodes; idx++) {
        ggml_tensor * op = ggml_graph_முனை(cgraph, idx);
        எனில் (dev->iface.supports_op(dev, op)) {
            தொடர்;
        }
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Graph node %d (%s) not supported by the backend\n", __func__, idx,
                       ggml_op_desc(op));

        status = GGML_STATUS_ABORTED;
        apir_encode_ggml_status(enc, &status);

        திரும்பு 0;
    }
#endif

    // Check if backend is properly initialized
    எனில் (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Backend not initialized (bck is null)\n", __func__);

        திரும்பு 1;
    }

    status = bck->iface.graph_compute(bck, cgraph);

    எனில் (async_backend && bck->iface.synchronize) {
        bck->iface.synchronize(bck);
    }

    apir_encode_ggml_status(enc, &status);

    திரும்பு 0;
}
