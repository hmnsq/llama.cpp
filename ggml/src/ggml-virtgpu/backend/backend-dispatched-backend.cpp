#ಅನ್ತರ್ಭಾವ "backend-dispatched.h"
#ಅನ್ತರ್ಭಾವ "backend-virgl-apir.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "shared/apir_backend.h"

#ಅನ್ತರ್ಭಾವ <cstdint>

ಸ್ಥಿರ uint32_t validate_graph_operation(size_t cgraph_size, uint32_t shmem_res_id, ಧ್ರುವ ವರ್ಣ * operation) {
    ಯದಿ (cgraph_size == 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Zero-size computation graph\n", operation);
        ಪ್ರತ್ಯಾಗಮ 1;
    }

    // place-holder: validate that the size of shmem_res_id is <= cgraph_size
    // need to add another method in the Virgl->APIR callback interface
    GGML_UNUSED(shmem_res_id);

    ಪ್ರತ್ಯಾಗಮ 0;  // Valid
}

uint32_t backend_backend_graph_compute(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx) {
    GGML_UNUSED(ctx);

    ಸ್ಥಿರ ತಾರ್ಕಿಕ async_backend_initialized = ಅಸತ್ಯ;
    ಸ್ಥಿರ ತಾರ್ಕಿಕ async_backend;

    ಯದಿ (!async_backend_initialized) {
        ggml_backend_dev_props props;

        dev->iface.get_props(dev, &props);
        async_backend             = props.caps.async;
        async_backend_initialized = ಸತ್ಯ;
    }

    uint32_t shmem_res_id;
    apir_decode_virtgpu_shmem_res_id(dec, &shmem_res_id);

    ಧ್ರುವ ಶೂನ್ಯ * shmem_data = ctx->iface->get_shmem_ptr(ctx->ctx_id, shmem_res_id);
    ಯದಿ (!shmem_data) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Couldn't get the shmem addr from virgl\n", __func__);
        apir_decoder_set_fatal(dec);
        ಪ್ರತ್ಯಾಗಮ 1;
    }
    size_t cgraph_size;
    apir_decode_size_t(dec, &cgraph_size);

    ಯದಿ (validate_graph_operation(cgraph_size, shmem_res_id, __func__) != 0) {
        apir_decoder_set_fatal(dec);
        ಪ್ರತ್ಯಾಗಮ 1;
    }

    apir_decoder secondary_dec = apir_new_decoder((ಧ್ರುವ ವರ್ಣ *) shmem_data, cgraph_size);

    ggml_cgraph * cgraph = apir_decode_ggml_cgraph(&secondary_dec, cgraph_size);

    ಯದಿ (!cgraph || apir_decoder_get_fatal(&secondary_dec)) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Failed to deserialize computation graph\n", __func__);
        ಪ್ರತ್ಯಾಗಮ 1;
    }

    ಯದಿ (cgraph->n_nodes < 0 || cgraph->n_leafs < 0) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Invalid negative node/leaf count: nodes=%d leafs=%d\n", __func__,
                       cgraph->n_nodes, cgraph->n_leafs);
        ಪ್ರತ್ಯಾಗಮ 1;
    }

    ggml_status status;
#ಯದಿ APIR_BACKEND_CHECK_SUPPORTS_OP == 1
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ idx = 0; idx < cgraph->n_nodes; idx++) {
        ggml_tensor * op = ggml_graph_ಗಂಟು(cgraph, idx);
        ಯದಿ (dev->iface.supports_op(dev, op)) {
            ಅನುವರ್ತ;
        }
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Graph node %d (%s) not supported by the backend\n", __func__, idx,
                       ggml_op_desc(op));

        status = GGML_STATUS_ABORTED;
        apir_encode_ggml_status(enc, &status);

        ಪ್ರತ್ಯಾಗಮ 0;
    }
#endif

    // Check if backend is properly initialized
    ಯದಿ (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Backend not initialized (bck is null)\n", __func__);

        ಪ್ರತ್ಯಾಗಮ 1;
    }

    status = bck->iface.graph_compute(bck, cgraph);

    ಯದಿ (async_backend && bck->iface.synchronize) {
        bck->iface.synchronize(bck);
    }

    apir_encode_ggml_status(enc, &status);

    ಪ್ರತ್ಯಾಗಮ 0;
}
