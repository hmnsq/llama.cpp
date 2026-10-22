#అన్తర్భావ "backend-dispatched.h"
#అన్తర్భావ "backend-virgl-apir.h"
#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "shared/apir_backend.h"

#అన్తర్భావ <cstdint>

స్థిర uint32_t validate_లేఖాచిత్రం_operation(size_t cgraph_కొలత, uint32_t shmem_res_id, స్థిరరాశి అక్షరం * operation) {
    ఒకవేళ (cgraph_కొలత == 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Zero-size computation graph\n", operation);
        తిరిగి 1;
    }

    // place-holder: validate that the size of shmem_res_id is <= cgraph_size
    // need to add another method in the Virgl->APIR callback interface
    GGML_వాడని(shmem_res_id);

    తిరిగి 0;  // Valid
}

uint32_t backend_backend_లేఖాచిత్రం_compute(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_వాడని(ctx);

    స్థిర తార్కిక async_backend_initialized = అబద్ధం;
    స్థిర తార్కిక async_backend;

    ఒకవేళ (!async_backend_initialized) {
        ggml_backend_dev_props props;

        dev->iface.get_props(dev, &props);
        async_backend             = props.caps.async;
        async_backend_initialized = నిజం;
    }

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    స్థిరరాశి శూన్య * shmem_దత్తాంశం = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    ఒకవేళ (!shmem_దత్తాంశం) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_సమితి_fatal(dec);
        తిరిగి 1;
    }
    size_t cgraph_కొలత;
    apir_decode_కొలత_t(dec, &cgraph_కొలత);

    ఒకవేళ (validate_లేఖాచిత్రం_operation(cgraph_కొలత, shmem_res_id, __func__) != 0) {
        apir_decoder_సమితి_fatal(dec);
        తిరిగి 1;
    }

    apir_decoder secondary_dec = apir_new_decoder((స్థిరరాశి అక్షరం *) shmem_దత్తాంశం, cgraph_కొలత);

    ggml_cgraph * cgraph = apir_decode_ggml_cgraph(&secondary_dec, cgraph_కొలత);

    ఒకవేళ (!cgraph || apir_decoder_get_fatal(&secondary_dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to deserialize computation graph\n", __func__);
        తిరిగి 1;
    }

    ఒకవేళ (cgraph->n_nodes < 0 || cgraph->n_leafs < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid negative node/leaf count: nodes=%d leafs=%d\n", __func__,
                       cgraph->n_nodes, cgraph->n_leafs);
        తిరిగి 1;
    }

    ggml_స్థితి స్థితి_2;
#ఒకవేళ APIR_BACKEND_CHECK_SUPPORTS_OP == 1
    కొరకు (పూర్ణాంకం idx = 0; idx < cgraph->n_nodes; idx++) {
        ggml_tensor * op = ggml_లేఖాచిత్రం_కణుపు(cgraph, idx);
        ఒకవేళ (dev->iface.supports_op(dev, op)) {
            కొనసాగు;
        }
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Graph node %d (%s) not supported by the backend\n", __func__, idx,
                       ggml_op_desc(op));

        స్థితి_2 = GGML_స్థితి_ABORTED;
        apir_encode_ggml_స్థితి(enc, &స్థితి_2);

        తిరిగి 0;
    }
#అయితే_ముగింపు

    // Check if backend is properly initialized
    ఒకవేళ (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Backend not initialized (bck is null)\n", __func__);

        తిరిగి 1;
    }

    స్థితి_2 = bck->iface.లేఖాచిత్రం_compute(bck, cgraph);

    ఒకవేళ (async_backend && bck->iface.synchronize) {
        bck->iface.synchronize(bck);
    }

    apir_encode_ggml_స్థితి(enc, &స్థితి_2);

    తిరిగి 0;
}
