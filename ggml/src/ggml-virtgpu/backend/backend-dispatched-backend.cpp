#અન્તર્ભાવ "backend-dispatched.h"
#અન્તર્ભાવ "backend-virgl-apir.h"
#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-backend.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "shared/apir_backend.h"

#અન્તર્ભાવ <cstdint>

સ્થિર uint32_t validate_graph_operation(size_t cgraph_size, uint32_t shmem_res_id, અચલ અક્ષર * operation) {
    જો (cgraph_size == 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Zero-size computation graph\n", operation);
        પાછા 1;
    }

    // place-holder: validate that the size of shmem_res_id is <= cgraph_size
    // need to add another method in the Virgl->APIR callback interface
    GGML_UNUSED(shmem_res_id);

    પાછા 0;  // Valid
}

uint32_t backend_backend_graph_compute(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);

    સ્થિર તાર્કિક async_backend_initialized = ખોટું;
    સ્થિર તાર્કિક async_backend;

    જો (!async_backend_initialized) {
        ggml_backend_dev_props props;

        dev->iface.get_props(dev, &props);
        async_backend             = props.caps.async;
        async_backend_initialized = સાચું;
    }

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    અચલ શૂન્ય * shmem_data = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    જો (!shmem_data) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_set_fatal(dec);
        પાછા 1;
    }
    size_t cgraph_size;
    apir_decode_size_t(dec, &cgraph_size);

    જો (validate_graph_operation(cgraph_size, shmem_res_id, __func__) != 0) {
        apir_decoder_set_fatal(dec);
        પાછા 1;
    }

    apir_decoder secondary_dec = apir_new_decoder((અચલ અક્ષર *) shmem_data, cgraph_size);

    ggml_cgraph * cgraph = apir_decode_ggml_cgraph(&secondary_dec, cgraph_size);

    જો (!cgraph || apir_decoder_get_fatal(&secondary_dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to deserialize computation graph\n", __func__);
        પાછા 1;
    }

    જો (cgraph->n_nodes < 0 || cgraph->n_leafs < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid negative node/leaf count: nodes=%d leafs=%d\n", __func__,
                       cgraph->n_nodes, cgraph->n_leafs);
        પાછા 1;
    }

    ggml_status status;
#જો APIR_BACKEND_CHECK_SUPPORTS_OP == 1
    માટે (પૂર્ણાંક idx = 0; idx < cgraph->n_nodes; idx++) {
        ggml_tensor * op = ggml_graph_ગાંઠ(cgraph, idx);
        જો (dev->iface.supports_op(dev, op)) {
            ચાલુ;
        }
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Graph node %d (%s) not supported by the backend\n", __func__, idx,
                       ggml_op_desc(op));

        status = GGML_STATUS_ABORTED;
        apir_encode_ggml_status(enc, &status);

        પાછા 0;
    }
#endif

    // Check if backend is properly initialized
    જો (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Backend not initialized (bck is null)\n", __func__);

        પાછા 1;
    }

    status = bck->iface.graph_compute(bck, cgraph);

    જો (async_backend && bck->iface.synchronize) {
        bck->iface.synchronize(bck);
    }

    apir_encode_ggml_status(enc, &status);

    પાછા 0;
}
